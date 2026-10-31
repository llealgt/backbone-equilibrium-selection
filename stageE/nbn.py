"""Neural equilibrium-selection harness (autograd, CPU). Exact counterfactual-value
regularized policy gradient (R-NaD / NashPG) with table, MLP, or attention policies."""
import numpy as np
import autograd.numpy as anp
from autograd import grad
import efg

# ---------------- reach probabilities (for reach-weighted objective) ----------
def reach_probs(game, sigma):
    w = {I: 0.0 for I in game.infosets()}
    def rec(node, r):
        if game.is_terminal(node): return
        I = game.infoset(node); w[I] += r; s = sigma[I]
        for a in game.actions(node): rec(game.next_node(node, a), r * s[a])
    for prob, root in game.roots(): rec(root, prob)
    return w

# ---------------- tokenizer: infoset -> padded token features ------------------
def build_features(game, typed=True):
    infos = game.infosets()
    I_list = list(infos.keys())
    nacts = {I: infos[I][1] for I in I_list}
    players = {I: infos[I][0] for I in I_list}
    # token = onehot(type:3) + onehot(value:VMAX)   (typed=False zeros the type block)
    T_TYPES = 3   # 0=player, 1=card/private, 2=history-action
    VMAX = 4
    D = T_TYPES + VMAX
    def tok(t, v):
        z = np.zeros(D)
        if typed: z[t] = 1.0
        z[T_TYPES + min(v, VMAX - 1)] = 1.0; return z
    def tokens_for(I):
        if isinstance(I, str):                      # matrix game "P0"/"P1"
            p = 0 if I == "P0" else 1
            return [tok(0, p)]
        p, card, hist = I
        ts = [tok(0, p), tok(1, card)]
        for a in hist: ts.append(tok(2, a))
        return ts
    toklists = [tokens_for(I) for I in I_list]
    maxT = max(len(t) for t in toklists)
    maxA = max(nacts.values())
    N = len(I_list)
    X = np.zeros((N, maxT, D)); M = np.zeros((N, maxT)); AM = np.zeros((N, maxA))
    for i, (ts, I) in enumerate(zip(toklists, I_list)):
        for j, t in enumerate(ts): X[i, j] = t; M[i, j] = 1.0
        AM[i, :nacts[I]] = 1.0
    return dict(I_list=I_list, nacts=nacts, players=players, X=X, M=M, AM=AM,
                D=D, maxT=maxT, maxA=maxA, N=N)

# ---------------- parameter init ----------------------------------------------
def init_params(arch, F, h=16, d=12, seed=0, typed=True, ln=False):
    """`ln` adds LayerNorm scale/shift parameters. LayerNorm is a Stage-A
    addition: the stability gap between backbones may be a normalization
    artifact rather than an architecture property, and that hypothesis has to be
    tested inside this notebook rather than assumed. It is available for BOTH
    backbones so that any LayerNorm effect is not confounded with architecture."""
    rng = np.random.default_rng(seed)
    D, maxT, maxA = F["D"], F["maxT"], F["maxA"]
    def r(*s, sc=0.3): return rng.standard_normal(s) * sc
    if arch == "table":
        return {"logits": r(F["N"], maxA, sc=0.01)}
    if arch == "mlp":
        din = D * maxT
        p = {"W1": r(din, h), "b1": np.zeros(h), "W2": r(h, h), "b2": np.zeros(h),
             "W3": r(h, maxA), "b3": np.zeros(maxA)}
        if ln:
            p.update({"g1": np.ones(h), "s1": np.zeros(h),
                      "g2": np.ones(h), "s2": np.zeros(h)})
        return p
    if arch == "attn":
        p = {"E": r(D, d), "Wq": r(d, d), "Wk": r(d, d), "Wv": r(d, d),
             "Wo": r(d, d), "W1": r(d, h), "b1": np.zeros(h),
             "W2": r(h, maxA), "b2": np.zeros(maxA)}
        if ln:
            p.update({"gA": np.ones(d), "sA": np.zeros(d),
                      "gP": np.ones(d), "sP": np.zeros(d)})
        return p
    raise ValueError(arch)


def sinusoidal_pe(maxT, d):
    """Fixed sinusoidal positional encoding (no learned parameters, so enabling
    it does not change parameter counts).

    Motivation: masked mean-pooling after self-attention is permutation
    invariant, so without positional information the attention policy is
    provably blind to the ORDER of history tokens while the MLP -- which
    flattens the sequence -- is not. The two backbones therefore do not consume
    equivalent information by default. `pe=True` removes that asymmetry."""
    pos = np.arange(maxT)[:, None]
    i = np.arange(d)[None, :]
    ang = pos / np.power(10000.0, (2 * (i // 2)) / max(d, 1))
    pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
    return pe

def count_params(p):
    return int(sum(np.asarray(v).size for v in p.values()))

# ---------------- forward passes (autograd) -----------------------------------
def _layernorm(x, g, b, eps=1e-5):
    mu = anp.mean(x, axis=-1, keepdims=True)
    var = anp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) / anp.sqrt(var + eps) * g + b


def _masked_softmax(logits, amask):
    logits = logits - anp.max(logits, axis=1, keepdims=True)
    e = anp.exp(logits) * amask
    return e / anp.sum(e, axis=1, keepdims=True)

def forward(arch, p, F, soft=True, causal=False, ln=False, pe=False):
    X, M, AM = F["X"], F["M"], F["AM"]
    N, maxT, D = X.shape
    if arch == "table":
        return _masked_softmax(p["logits"], AM)
    if arch == "mlp":
        flat = X.reshape(N, maxT * D)
        z1 = flat @ p["W1"] + p["b1"]
        if ln: z1 = _layernorm(z1, p["g1"], p["s1"])
        h1 = anp.tanh(z1) if soft else anp.maximum(z1, 0)
        z2 = h1 @ p["W2"] + p["b2"]
        if ln: z2 = _layernorm(z2, p["g2"], p["s2"])
        h2 = anp.tanh(z2) if soft else anp.maximum(z2, 0)
        return _masked_softmax(h2 @ p["W3"] + p["b3"], AM)
    if arch == "attn":
        emb = X @ p["E"]                                  # [N,T,d]
        if pe:
            emb = emb + F.get("PE", sinusoidal_pe(X.shape[1], emb.shape[-1]))[None]
        Q = emb @ p["Wq"]; K = emb @ p["Wk"]; V = emb @ p["Wv"]
        d = emb.shape[-1]
        scores = anp.einsum("ntd,nsd->nts", Q, K) / np.sqrt(d)  # [N,T,T]
        big = (1.0 - M[:, None, :]) * (-1e9)              # key padding mask
        scores = scores + big
        if causal:
            cm = np.triu(np.ones((maxT, maxT)), 1)[None] * (-1e9)
            scores = scores + cm
        if soft:
            scores = scores - anp.max(scores, axis=2, keepdims=True)
            A = anp.exp(scores); A = A / anp.sum(A, axis=2, keepdims=True)
        else:                                            # hard-ish attention
            scores = scores * 12.0
            scores = scores - anp.max(scores, axis=2, keepdims=True)
            A = anp.exp(scores); A = A / anp.sum(A, axis=2, keepdims=True)
        ctx = anp.einsum("nts,nsd->ntd", A, V) @ p["Wo"] + emb   # residual
        if ln: ctx = _layernorm(ctx, p["gA"], p["sA"])             # post-attention LN
        mm = M[:, :, None]
        pooled = anp.sum(ctx * mm, axis=1) / anp.sum(mm, axis=1)  # masked mean
        if ln: pooled = _layernorm(pooled, p["gP"], p["sP"])
        z1 = pooled @ p["W1"] + p["b1"]
        h1 = anp.tanh(z1) if soft else anp.maximum(z1, 0)
        return _masked_softmax(h1 @ p["W2"] + p["b2"], AM)
    raise ValueError(arch)

# ---------------- sigma dict <-> rows -----------------------------------------
def rows_to_sigma(rows, F):
    sig = {}
    for i, I in enumerate(F["I_list"]):
        n = F["nacts"][I]; sig[I] = np.array(rows[i, :n], float)
    return sig

# ---------------- Adam ---------------------------------------------------------
class Adam:
    def __init__(self, p, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}; self.t = 0
    def step(self, p, g):
        self.t += 1
        for k in p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            p[k] = p[k] + self.lr * mh / (np.sqrt(vh) + self.eps)   # ascent
        return p

# ---------------- trainer: fitted-target neural R-NaD / NashPG -----------------
def train(game, arch="mlp", eta=0.5, lr_q=1.0, opt_lr=0.05, iters=900,
          ref_period=30, inner=4, solver="rnad", seed=0, h=16, d=12, soft=True,
          causal=False, ref_kind="uniform", track=True, F=None, params=None,
          return_best=True, grad_clip=10.0, early_stop=None, eval_every=None,
          ln=False, pe=False, exact_nc=True):
    """Fitted-target form: each outer step computes the exact tabular R-NaD softmax
    target tau_I from exact counterfactual values, then fits the network to tau by
    reach-weighted KL for `inner` gradient steps. In the table limit this reproduces
    tabular R-NaD exactly; with a shared network the inductive bias enters here.
    solver 'rnad' resets the reference to the current policy; 'nashpg' also refines
    the reference but is meant to be run with a large fixed eta (no annealing).

    Robustness (so long runs cannot be corrupted by post-convergence divergence):
      * return_best : report the lowest-exploitability iterate seen, not the final one.
      * grad_clip   : clip the global gradient norm each inner step (prevents blow-up).
      * early_stop  : optional exploitability tolerance; stop once best < tol.
    The returned dict also carries `final_exploit` (last-iterate exploitability) so
    stability — did the run *stay* converged — can be judged separately from selection."""
    if F is None: F = build_features(game)
    if params is None: params = init_params(arch, F, h=h, d=d, seed=seed, ln=ln)
    if pe and "PE" not in F:
        F = dict(F); F["PE"] = sinusoidal_pe(F["maxT"], d)
    nc = efg.nash_conv_dp if exact_nc else efg.nash_conv
    fwd = lambda p_: forward(arch, p_, F, soft=soft, causal=causal, ln=ln, pe=pe)
    opt = Adam(params, lr=opt_lr)
    AM = F["AM"]
    if ref_kind == "uniform":
        ref_rows = np.array(AM / AM.sum(1, keepdims=True))
    else:
        rng = np.random.default_rng(seed + 777)
        ref_rows = np.array(_masked_softmax(rng.standard_normal(AM.shape), AM))
    logref = np.log(ref_rows + 1e-12)
    c = 1.0 / (1.0 + lr_q * eta)
    cad = eval_every or max(1, iters // 60)
    traj, gnorms = [], []
    best = {"exploit": np.inf}
    numeric_ok = True
    def snap(t, sig):
        ex = float(nc(game, sig)); He = float(efg.mean_entropy(game, sig))
        co = float(game.selection_coord(sig)); traj.append((t, ex, He, co))
        if ex < best["exploit"]:
            best.update(exploit=ex, entropy=He, coord=co, iter=t,
                        params={k: np.array(v).copy() for k, v in params.items()},
                        sigma={I: np.array(sig[I]).copy() for I in sig})
        return ex
    for t in range(1, iters + 1):
      with np.errstate(over="ignore", invalid="ignore"):
        rows = np.array(fwd(params))
        if not np.all(np.isfinite(rows)):        # numerical-validity guard
            numeric_ok = False
            break
        sig = rows_to_sigma(rows, F)
        if track and (t % cad == 0 or t == 1 or t == iters):
            snap(t, sig)
            if early_stop is not None and best["exploit"] < early_stop: break
        q, reach = efg.cf_values(game, sig); w = reach_probs(game, sig)
        # exact tabular R-NaD target tau and reach weights
        logsig = np.log(rows + 1e-12); QA = np.zeros_like(rows); W = np.zeros(F["N"])
        for i, I in enumerate(F["I_list"]):
            n = F["nacts"][I]; rr = reach[I] if reach[I] > 1e-12 else 1.0
            QA[i, :n] = q[I] / rr; W[i] = w[I]
        tlog = c * logsig + c * lr_q * QA + (1 - c) * logref
        tau = np.array(_masked_softmax(tlog, AM)); Wn = W / (W.sum() + 1e-12)
        # fit network to tau: minimize sum_I Wn_I * KL(tau_I || sigma_theta_I)
        def loss(p):
            s = fwd(p)
            ce = -anp.sum(tau * anp.log(s + 1e-12), axis=1)
            return anp.sum(Wn * ce)
        for _ in range(inner):
            g = grad(loss)(params)
            gn = float(np.sqrt(sum(np.sum(np.asarray(v) ** 2) for v in g.values())))
            gnorms.append(gn)
            if grad_clip and gn > grad_clip:                 # global-norm clip
                sc = grad_clip / (gn + 1e-12); g = {k: g[k] * sc for k in g}
            params = opt.step(params, {k: -g[k] for k in g})   # descent on loss
        if t % ref_period == 0:
            rows2 = np.array(fwd(params))
            ref_rows = rows2; logref = np.log(ref_rows + 1e-12)
    with np.errstate(over="ignore", invalid="ignore"):
        rows = np.array(fwd(params))
    if not np.all(np.isfinite(rows)):
        numeric_ok = False
        rows = np.array(AM / AM.sum(1, keepdims=True))
    sig_f = rows_to_sigma(rows, F)
    final_ex = float(nc(game, sig_f))
    if return_best and "params" in best:
        sig, params, ex, He, co, bit = (best["sigma"], best["params"], best["exploit"],
                                        best["entropy"], best["coord"], best["iter"])
    else:
        sig, ex = sig_f, final_ex
        He = float(efg.mean_entropy(game, sig_f)); co = float(game.selection_coord(sig_f)); bit = iters
    # ---- final-iterate and late-window readings -----------------------------
    # Reported alongside the best iterate so the optimistic bias of best-of-N
    # sampling can be measured per architecture rather than asserted. `best` is
    # the minimum over ~60 sampled evaluations; min-of-N is biased low, and more
    # so for noisier runs, which is exactly the arm under test.
    coord_final = float(game.selection_coord(sig_f))
    entropy_final = float(efg.mean_entropy(game, sig_f))
    T = np.array(traj)
    if len(T):
        k = max(1, len(T) // 5)                       # last 20% of evaluations
        coord_late = float(np.mean(T[-k:, 3]))
        exploit_late = float(np.mean(T[-k:, 1]))
        entropy_late = float(np.mean(T[-k:, 2]))
    else:
        coord_late, exploit_late, entropy_late = coord_final, final_ex, entropy_final
    return dict(sigma=sig, params=params, F=F, traj=T,
                grad_var=float(np.var(gnorms[len(gnorms)//2:])) if gnorms else 0.0,
                grad_mean=float(np.mean(gnorms)) if gnorms else 0.0,
                exploit=ex, entropy=He, coord=co, nparams=count_params(params),
                final_exploit=final_ex, best_iter=bit,
                coord_final=coord_final, entropy_final=entropy_final,
                coord_late=coord_late, exploit_late=exploit_late,
                entropy_late=entropy_late, numeric_ok=bool(numeric_ok))

if __name__ == "__main__":
    import time
    g = efg.GAMES["kuhn"]; F = build_features(g)
    print("Kuhn infosets:", F["N"], "maxT", F["maxT"], "maxA", F["maxA"])
    for arch, kw in [("table", {}), ("mlp", dict(h=16)), ("attn", dict(h=16, d=12))]:
        t0 = time.time()
        r = train(g, arch=arch, eta=0.5, iters=1200, seed=0, F=F, **kw)
        print(f"{arch:6s} params={r['nparams']:4d} exploit={r['exploit']:.4f} "
              f"H={r['entropy']:.4f} coord={r['coord']:.4f} time={time.time()-t0:.1f}s")

# ---------------- ground truth + robustness helpers ---------------------------
def ground_truth(game, fam, lo, hi, n=801):
    cs = np.linspace(lo, hi, n); H = np.array([efg.mean_entropy(game, fam(c)) for c in cs])
    i = int(np.argmax(H)); cstar = cs[i]; Hmax = H[i]
    m = np.abs(cs - cstar) <= (hi - lo) * 0.10
    a = np.polyfit(cs[m], H[m], 2); kappa = -2 * a[0]
    return dict(cstar=float(cstar), Hmax=float(Hmax), kappa=float(kappa))

def delta_tabular(game, eta, Hmax):
    s = efg.train(game, iters=15000, updater="rnad", lr=1.0, alpha=eta,
                  ref_period=60, track_every=99999)["sigma"]
    return Hmax - efg.mean_entropy(game, s), float(game.selection_coord(s)), float(efg.nash_conv(game, s))

def kuhn_fixed_opponents():
    """Fixed P1 strategies for off-equilibrium robustness on Kuhn."""
    def base():
        return {(1, 0, (0,)): np.array([1.0, 0.0]), (1, 1, (0,)): np.array([1.0, 0.0]),
                (1, 2, (0,)): np.array([1.0, 0.0]), (1, 0, (1,)): np.array([1.0, 0.0]),
                (1, 1, (1,)): np.array([1.0, 0.0]), (1, 2, (1,)): np.array([1.0, 0.0])}
    over_fold = base()                                   # always check / fold
    over_call = {I: np.array([0.0, 1.0]) for I in base()}  # always bet / call
    rng = np.random.default_rng(0)
    rand = {I: (lambda p: p / p.sum())(rng.random(2)) for I in base()}
    return {"over_fold": over_fold, "over_call": over_call, "random": rand}

def p0_value_vs(game, p0_sigma, p1_sigma):
    s = dict(p0_sigma); s.update(p1_sigma)
    return float(efg.value(game, s))

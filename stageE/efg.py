"""
Generic tabular extensive-form engine for the equilibrium-selection study.

The SAME solver code (CFR, CFR+, R-NaD, MMD) runs on every game, so any
difference in the equilibrium selected is a property of the algorithm, not of
bespoke per-game code. Each game is a small object exposing a uniform protocol
and, crucially, a `selection_coord(sigma)` that returns the free parameter of
its (analytically known) Nash set, plus the max-entropy member of that set.
"""
import numpy as np, itertools

def _softmax(v):
    v = v - v.max(); e = np.exp(v); return e / e.sum()

# ---------------------------------------------------------------- Game protocol
class Game:
    """Subclasses define: roots(), is_terminal, util, player, infoset, actions, next_node."""
    def infosets(self):
        seen = {}
        def rec(node):
            if self.is_terminal(node): return
            I = self.infoset(node); seen[I] = (self.player(node), len(self.actions(node)))
            for a in self.actions(node): rec(self.next_node(node, a))
        for _, r in self.roots(): rec(r)
        return seen
    def player_infosets(self, p):
        return [I for I, (pl, _) in self.infosets().items() if pl == p]

# ---------------------------------------------------------------- core routines
def value(game, sigma, node=None, _root=True):
    """Expected P0 utility under sigma."""
    if _root:
        return sum(prob * value(game, sigma, r, _root=False) for prob, r in game.roots())
    if game.is_terminal(node): return game.util(node)
    I = game.infoset(node); s = sigma[I]
    return sum(s[a] * value(game, sigma, game.next_node(node, a), _root=False)
               for a in game.actions(node))

def cf_values(game, sigma):
    """Per infoset I of its acting player p: (q, reach) with qa=q/reach the
    counterfactual-reach-weighted average action value, in p's own units."""
    infos = game.infosets()
    q = {I: np.zeros(n) for I, (_, n) in infos.items()}
    reach = {I: 0.0 for I in infos}
    def rec(node, rp0, rp1, rc):
        if game.is_terminal(node): return game.util(node)
        p = game.player(node); I = game.infoset(node); acts = game.actions(node); s = sigma[I]
        opp = (rp1 * rc) if p == 0 else (rp0 * rc)
        cvs = []
        for a in acts:
            child = game.next_node(node, a)
            cvs.append(rec(child, rp0 * s[a], rp1, rc) if p == 0 else rec(child, rp0, rp1 * s[a], rc))
        sign = 1.0 if p == 0 else -1.0
        for a in acts: q[I][a] += opp * sign * cvs[a]
        reach[I] += opp
        return sum(s[a] * cvs[a] for a in acts)
    for prob, r in game.roots(): rec(r, 1.0, 1.0, prob)
    return q, reach

def best_response_strategy(game, sigma, br_player):
    """Like best_response_value but returns the argmax PURE strategy (dict of
    one-hot vectors over br_player's infosets)."""
    infos = game.player_infosets(br_player)
    sizes = [len(game.actions_for_infoset(I)) for I in infos]
    best, arg = -1e9, None
    for combo in itertools.product(*[range(k) for k in sizes]):
        cand = {I: sigma[I] for I in sigma}
        for I, a in zip(infos, combo):
            v = np.zeros(len(game.actions_for_infoset(I))); v[a] = 1.0; cand[I] = v
        own = (value(game, cand)) * (1 if br_player == 0 else -1)
        if own > best:
            best = own; arg = {I: cand[I].copy() for I in infos}
    return arg

def _own_reach(game, pure, p):
    """Realization probability of each of player p's infosets under p's own
    (pure) strategy -- needed to average behavioural strategies correctly in
    extensive form."""
    reach = {I: 0.0 for I in game.player_infosets(p)}
    def rec(node, r):
        if game.is_terminal(node):
            return
        I = game.infoset(node)
        if game.player(node) == p:
            reach[I] = r                                   # identical across histories (perfect recall)
            for a in game.actions(node):
                rec(game.next_node(node, a), r * pure[I][a])
        else:
            for a in game.actions(node):
                rec(game.next_node(node, a), r)
    for _, root in game.roots():
        rec(root, 1.0)
    return reach

def fictitious_play(game, iters=3000, init=None):
    """Realization-weighted fictitious play: each player best-responds to the
    opponent's current average; averages accumulate in behavioural form."""
    infos = game.infosets()
    cum = {I: np.zeros(n) for I, (_, n) in infos.items()}
    w = {I: 0.0 for I in infos}
    avg = uniform(game) if init is None else {I: np.array(init[I], float) for I in infos}
    for t in range(iters):
        for p in (0, 1):
            br = best_response_strategy(game, avg, p)
            rch = _own_reach(game, br, p)
            for I in game.player_infosets(p):
                cum[I] += rch[I] * br[I]; w[I] += rch[I]
        for I in infos:
            avg[I] = cum[I] / w[I] if w[I] > 1e-12 else uniform(game)[I]
    return {"sigma": avg, "avg_sigma": avg}

# ---- geometric metrics (game-agnostic) ----
def _kl(p, q):
    p = p + 1e-12; q = q + 1e-12
    return float((p * np.log(p / q)).sum())

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def profile_js(game, sa, sb):
    """Mean Jensen-Shannon divergence between two strategy profiles across infosets."""
    return float(np.mean([js_divergence(sa[I], sb[I]) for I in game.infosets()]))

def l0_support(game, sigma, thresh=1e-4):
    """Average number of actions played with probability above threshold."""
    return float(np.mean([(sigma[I] > thresh).sum() for I in game.infosets()]))

def best_response_value(game, sigma, br_player):
    """br_player's best-response utility (own units) vs fixed opponent, by pure
    strategy enumeration (games here are tiny)."""
    infos = game.player_infosets(br_player)
    sizes = [len(game.actions_for_infoset(I)) for I in infos]
    best = -1e9
    for combo in itertools.product(*[range(k) for k in sizes]):
        cand = {I: sigma[I] for I in sigma}
        for I, a in zip(infos, combo):
            v = np.zeros(len(game.actions_for_infoset(I))); v[a] = 1.0; cand[I] = v
        val = value(game, cand)
        own = val if br_player == 0 else -val
        if own > best: best = own
    return best

def nash_conv(game, sigma):
    return best_response_value(game, sigma, 0) + best_response_value(game, sigma, 1)

def uniform(game):
    return {I: np.ones(n) / n for I, (_, n) in game.infosets().items()}

def train(game, iters=8000, updater="cfrplus", lr=1.0, alpha=0.5, ref_period=60,
          seed=0, track_every=99999, init=None, magnet=None):
    rng = np.random.default_rng(seed)
    infos = game.infosets()
    uni = uniform(game)
    sigma = uniform(game) if init is None else {I: np.array(init[I], float) for I in infos}
    mag = uni if magnet is None else {I: np.array(magnet[I], float) for I in infos}
    cum_regret = {I: np.zeros(n) for I, (_, n) in infos.items()}
    cum_strategy = {I: np.zeros(n) for I, (_, n) in infos.items()}
    ref = {I: sigma[I].copy() for I in sigma}
    hist_it, hist_x = [], []
    for t in range(1, iters + 1):
        q, reach = cf_values(game, sigma)
        for I in infos:
            qa = q[I] / reach[I] if reach[I] > 1e-12 else q[I]
            s = sigma[I]
            if updater == "cfr":
                v = s @ qa; cum_regret[I] += qa - v
                pos = np.maximum(cum_regret[I], 0); sigma[I] = pos / pos.sum() if pos.sum() > 0 else uni[I].copy()
                cum_strategy[I] += s
            elif updater == "cfrplus":
                v = s @ qa; cum_regret[I] = np.maximum(cum_regret[I] + (qa - v), 0)
                tot = cum_regret[I].sum(); sigma[I] = cum_regret[I] / tot if tot > 0 else uni[I].copy()
                cum_strategy[I] += t * s
            elif updater == "mmd":
                c = 1 / (1 + lr * alpha)
                sigma[I] = _softmax(c * np.log(s + 1e-12) + c * lr * qa + (1 - c) * np.log(mag[I] + 1e-12))
            elif updater == "hedge":
                v = s @ qa; cum_regret[I] += qa - v          # same regrets as CFR...
                sigma[I] = _softmax(alpha * cum_regret[I])    # ...but softmax instead of max(R,0)
                cum_strategy[I] += s
            elif updater == "mwu":
                sigma[I] = _softmax(np.log(s + 1e-12) + lr * qa)   # unregularized (= tabular NeuRD)
            elif updater == "rnad":
                c = 1 / (1 + lr * alpha)
                sigma[I] = _softmax(c * np.log(s + 1e-12) + c * lr * qa + (1 - c) * np.log(ref[I] + 1e-12))
        if updater == "rnad" and t % ref_period == 0:
            ref = {I: sigma[I].copy() for I in sigma}
        if t % track_every == 0:
            avg = _avg(cum_strategy, uni) if updater in ("cfr", "cfrplus", "hedge") else sigma
            hist_it.append(t); hist_x.append(nash_conv(game, avg))
    out = {"iters": np.array(hist_it), "exploit": np.array(hist_x), "sigma": sigma}
    if updater in ("cfr", "cfrplus", "hedge"): out["avg_sigma"] = _avg(cum_strategy, uni)
    return out

def _avg(cum_strategy, uni):
    avg = {}
    for I, c in cum_strategy.items():
        avg[I] = c / c.sum() if c.sum() > 0 else uni[I].copy()
    return avg

def solution(res):
    return res.get("avg_sigma", res["sigma"])

def mean_entropy(game, sigma):
    return float(np.mean([-(p * np.log(p + 1e-12)).sum() for p in sigma.values()]))

# =====================================================================  GAMES
class MatrixGame(Game):
    """P0 (row, max) and P1 (col, min, sees nothing) play a one-shot zero-sum
    matrix M (P0 payoff). Encoded as a 2-node EFG with one infoset per player."""
    def __init__(self, name, M, coord_fn, maxent_coord, coord_name, coord_range):
        self.name = name; self.M = np.asarray(M, float)
        self._coord = coord_fn; self.maxent = maxent_coord; self.coord_name = coord_name
        self.coord_range = coord_range
        self.nr, self.nc = self.M.shape
    def roots(self): return [(1.0, ())]
    def is_terminal(self, node): return len(node) == 2
    def util(self, node): return self.M[node[0], node[1]]
    def player(self, node): return 0 if len(node) == 0 else 1
    def infoset(self, node): return "P0" if len(node) == 0 else "P1"
    def actions(self, node): return list(range(self.nr if len(node) == 0 else self.nc))
    def actions_for_infoset(self, I): return list(range(self.nr if I == "P0" else self.nc))
    def next_node(self, node, a): return node + (a,)
    def selection_coord(self, sigma): return self._coord(sigma)

class KuhnGame(Game):
    DECK = [0, 1, 2]  # J,Q,K
    name = "kuhn"; coord_name = "bluff freq P(bet|Jack)"; maxent = None; coord_range = (0.0, 1/3)
    def roots(self):
        deals = [(c0, c1) for c0 in self.DECK for c1 in self.DECK if c0 != c1]
        return [(1.0 / len(deals), (c0, c1, ())) for c0, c1 in deals]
    def is_terminal(self, node):
        h = node[2]
        return h in {(0, 0), (0, 1, 0), (0, 1, 1), (1, 0), (1, 1)}
    def util(self, node):
        c0, c1, h = node; win = 1 if c0 > c1 else -1
        if h == (0, 0): return win
        if h == (1, 0): return 1          # P0 bet, P1 fold
        if h == (1, 1): return 2 * win    # bet-call
        if h == (0, 1, 0): return -1      # P0 check, P1 bet, P0 fold
        if h == (0, 1, 1): return 2 * win # check-bet-call
    def player(self, node):
        h = node[2]; return 0 if len(h) in (0, 2) else 1
    def infoset(self, node):
        c0, c1, h = node; p = self.player(node)
        return (p, c0 if p == 0 else c1, h)
    def actions(self, node): return [0, 1]
    def actions_for_infoset(self, I): return [0, 1]
    def next_node(self, node, a):
        c0, c1, h = node; return (c0, c1, h + (a,))
    def selection_coord(self, sigma):  # P(bet | Jack at root)
        return sigma[(0, 0, ())][1]

# ---- matrix-game payoff matrices with analytically known Nash segments ----
def _pennies_safe():
    # rows r0,r1 = matching pennies; r2 = safe (always 0). P0 has a segment:
    # any (s2 mass on r2, rest split evenly r0/r1) is optimal; value 0.
    M = [[1, -1], [-1, 1], [0, 0]]
    coord = lambda sig: sig["P0"][2]          # mass on the safe action
    return MatrixGame("pennies_safe", M, coord, 1/3, "safe-action mass P(r2)", (0.0, 1.0))

def _dup_action():
    # cols c0,c1 identical => P1 segment: must put 1/2 on {c0,c1} (free split), 1/2 on c2.
    M = [[1, 1, -1], [-1, -1, 1]]
    coord = lambda sig: sig["P1"][0]          # split onto the first duplicate
    return MatrixGame("dup_action", M, coord, 1/4, "duplicate split P(c0)", (0.0, 0.5))

def _two_safe():
    # two safe rows (r2,r3) + matching pennies (r0,r1): 2-D optimal set; coord = total safe mass.
    M = [[1, -1], [-1, 1], [0, 0], [0, 0]]
    coord = lambda sig: sig["P0"][2] + sig["P0"][3]
    return MatrixGame("two_safe", M, coord, 0.5, "total safe mass P(r2)+P(r3)", (0.0, 1.0))

def kuhn_family_strategy(eta):
    """The analytic Kuhn player-0 Nash family, parameterised by eta=P(bet|Jack)
    in [0,1/3]; player-1 strategy is unique. Returns a full sigma dict."""
    s = {}
    # P0 root bets
    s[(0, 0, ())] = np.array([1 - eta, eta])       # Jack: bluff eta
    s[(0, 1, ())] = np.array([1.0, 0.0])           # Queen: never bet
    s[(0, 2, ())] = np.array([1 - 3 * eta, 3 * eta])  # King: value-bet 3*eta
    # P0 facing a bet after checking (call/fold)
    s[(0, 0, (0, 1))] = np.array([1.0, 0.0])       # Jack: fold
    s[(0, 1, (0, 1))] = np.array([2 / 3 - eta, 1 / 3 + eta])  # Queen: call eta+1/3
    s[(0, 2, (0, 1))] = np.array([0.0, 1.0])       # King: call
    # P1 after P0 checks (history (0,)): check/bet
    s[(1, 0, (0,))] = np.array([2 / 3, 1 / 3])     # Jack: bet 1/3
    s[(1, 1, (0,))] = np.array([1.0, 0.0])         # Queen: check
    s[(1, 2, (0,))] = np.array([0.0, 1.0])         # King: bet
    # P1 facing a bet (history (1,)): call/fold
    s[(1, 0, (1,))] = np.array([1.0, 0.0])         # Jack: fold
    s[(1, 1, (1,))] = np.array([2 / 3, 1 / 3])     # Queen: call 1/3
    s[(1, 2, (1,))] = np.array([0.0, 1.0])         # King: call
    return s

def kuhn_max_entropy_coord(grid=400):
    g = KuhnGame(); best = (-1, None)
    for eta in np.linspace(0, 1 / 3, grid):
        s = kuhn_family_strategy(eta)
        if mean_entropy(g, s) > best[0]: best = (mean_entropy(g, s), eta)
    return best[1]

def _asym_safe():
    # active rows r0,r1 form a SKEWED pennies (P1 must play 2:1, so the optimal
    # P0 segment is (p0, 2*p0, 1-3*p0) for p0 in [0,1/3]) — an *asymmetric* Nash
    # segment, unlike the symmetric games above. r2 is the value-matching safe row.
    M = [[3, -1], [-1, 1], [1 / 3, 1 / 3]]
    coord = lambda sig: sig["P0"][0]                       # mass on r0 (the free param)
    p0s = np.linspace(0, 1 / 3, 400)
    ent = lambda p: -sum(q * np.log(q + 1e-12) for q in [p, 2 * p, 1 - 3 * p])
    maxent = float(p0s[int(np.argmax([ent(p) for p in p0s]))])
    return MatrixGame("asym_safe", M, coord, maxent, "active mass P(r0)", (0.0, 1/3))

KuhnGame.maxent = kuhn_max_entropy_coord()

def _polytope4():
    # 2x4 skewed game. The MP part [[3,-1],[-1,1]] forces P1 to play 2:1, which
    # pins p1 = 2*p0; two value-matching safe rows r2,r3 absorb the remaining mass.
    # Nash set is a 2-D polytope: {(p0, 2*p0, p2, 1-3*p0-p2) : p0 in [0,1/3], p2>=0, 3p0+p2<=1}.
    M = [[3, -1], [-1, 1], [1 / 3, 1 / 3], [1 / 3, 1 / 3]]
    coord = lambda sig: sig["P0"][0]                       # primary (asymmetric) axis
    best = (-1, None)
    for p0 in np.linspace(0, 1 / 3, 160):
        for p2 in np.linspace(0, 1 - 3 * p0, 160):
            v = [p0, 2 * p0, p2, 1 - 3 * p0 - p2]
            if min(v) < -1e-9:
                continue
            H = -sum(q * np.log(q + 1e-12) for q in v)
            if H > best[0]:
                best = (H, (p0, p2))
    g = MatrixGame("polytope4", M, coord, best[1][0], "P(r0) [asym axis]", (0.0, 1 / 3))
    g.maxent2 = best[1]                                    # full 2-D max-entropy point (p0*, p2*)
    return g

GAMES = {g.name: g for g in [KuhnGame(), _pennies_safe(), _dup_action(),
                             _two_safe(), _asym_safe(), _polytope4()]}


# ============================================================================
# Dynamic-programming best response  (Stage A addition)
# ----------------------------------------------------------------------------
# `best_response_value` above enumerates every pure strategy via
# itertools.product over the player's infosets: 2**6 = 64 combinations on Kuhn,
# which is why it is fast there. On Leduc (~70 infosets per player) that is
# ~3**70 and the study cannot be extended. This DP computes the exact same
# quantity in a single pass per depth level.
#
# Method: process the player's infosets in order of decreasing depth. When
# resolving infoset I, every infoset strictly below it is already resolved, and
# counterfactual values depend only on the OPPONENT's and chance's reach, not
# on the player's own ancestor choices -- so one traversal per depth level
# yields exact counterfactual action values for every infoset at that level.
# Valid for perfect-recall trees, which is the class this engine builds.
# ============================================================================

def _infoset_depths(game):
    depth = {}
    def rec(node, d):
        if game.is_terminal(node):
            return
        I = game.infoset(node)
        depth[I] = max(depth.get(I, 0), d)
        for a in game.actions(node):
            rec(game.next_node(node, a), d + 1)
    for _, r in game.roots():
        rec(r, 0)
    return depth


def best_response_dp(game, sigma, br_player, return_value=False):
    """Exact best response for `br_player` against fixed `sigma`.

    Returns a dict of one-hot behavioural strategies over br_player's infosets
    (and, with return_value=True, the best-response value in br_player's own
    units). Equivalent to `best_response_strategy` / `best_response_value` but
    polynomial in the tree size instead of exponential in the infoset count.
    """
    infos = game.infosets()
    mine = [I for I, (p, _) in infos.items() if p == br_player]
    depth = _infoset_depths(game)
    sign = 1.0 if br_player == 0 else -1.0
    chosen = {}                                   # infoset -> action index

    levels = sorted({depth[I] for I in mine}, reverse=True)
    for lvl in levels:
        targets = {I for I in mine if depth[I] == lvl}
        cfv = {I: np.zeros(infos[I][1]) for I in targets}

        def rec(node, opp):
            """Return br_player's expected value below `node`; accumulate
            counterfactual action values for infosets in `targets`."""
            if game.is_terminal(node):
                return sign * game.util(node)
            p = game.player(node)
            I = game.infoset(node)
            acts = game.actions(node)
            if p == br_player:
                if I in chosen:                    # already resolved (deeper)
                    a = chosen[I]
                    return rec(game.next_node(node, a), opp)
                vals = [rec(game.next_node(node, a), opp) for a in acts]
                if I in targets:
                    for a in acts:
                        cfv[I][a] += opp * vals[a]
                return max(vals)                   # placeholder; only ancestors read it
            s = sigma[I]
            return sum(s[a] * rec(game.next_node(node, a), opp * s[a]) for a in acts)

        for prob, root in game.roots():
            rec(root, prob)
        for I in targets:
            chosen[I] = int(np.argmax(cfv[I]))

    br = {}
    for I in mine:
        v = np.zeros(infos[I][1])
        v[chosen[I]] = 1.0
        br[I] = v
    if not return_value:
        return br
    cand = {I: sigma[I] for I in sigma}
    cand.update(br)
    val = value(game, cand)
    return br, float(val if br_player == 0 else -val)


def best_response_value_dp(game, sigma, br_player):
    return best_response_dp(game, sigma, br_player, return_value=True)[1]


def nash_conv_dp(game, sigma):
    """Exact NashConv via DP best responses. Drop-in for `nash_conv`."""
    return (best_response_value_dp(game, sigma, 0)
            + best_response_value_dp(game, sigma, 1))

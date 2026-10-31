"""Stage-E registered analysis. Run once, after every registered job has finished:

    python analyzeE.py stageE_cache.json  > stageE_report.txt

Every test, threshold and decision rule here is part of the registration. The
script writes `stageE_results.json` next to the cache with every number it prints.

Conventions (identical to the notebook):
  * effect = mean(attention arm) - mean(MLP arm)  (selstats.welch(x=mlp, y=attn))
  * "read-off" coordinate = game.selection_coord(best-iterate policy), key "coord"
  * convergence gate: best-iterate exploitability < 0.02 and numeric_ok
"""
import json, os, sys
import numpy as np
from scipy import stats as st
import efg, nbn, games2
import selstats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "stageE_config.json")))
A = CFG["analysis"]
CONV = A["conv_thresh"]


# ------------------------------------------------------------------ data access
def values(C, block, game, arm, key="coord", gate="best"):
    """Gated per-seed values for one cell. gate: 'best' (registered primary) or 'final'."""
    blk = next(b for b in CFG["blocks"] if b["name"] == block)
    out = []
    for s in range(blk["seed_start"], blk["seed_start"] + blk["n_per_arm"]):
        r = C.get("%s|%s|%s|%d" % (block, game, arm, s))
        if r is None or key not in r or not r.get("numeric_ok", True):
            continue
        ex = r["exploit"] if gate == "best" else r["final_exploit"]
        if ex < CONV:
            out.append(r[key])
    return np.array(out, dtype=float)


def n_registered(block):
    return next(b for b in CFG["blocks"] if b["name"] == block)["n_per_arm"]


# ------------------------------------------------------------------ statistics
def pooled(effects, rho=0.0):
    """Fixed-effect inverse-variance pooling; `rho` = assumed equicorrelation of the
    per-game errors (0 = independence). Returns estimate, se, CI, two-sided p."""
    y = np.array([e["diff"] for e in effects]); s = np.array([e["se"] for e in effects])
    w = 1 / s**2; W = w.sum(); fe = float((w * y).sum() / W)
    cov = sum(w[i] * w[j] * rho * s[i] * s[j] for i in range(len(y)) for j in range(len(y)) if i != j)
    se = float(np.sqrt(((w**2 * s**2).sum() + cov) / W**2))
    z = fe / se
    return dict(est=fe, se=se, lo=fe - 1.96 * se, hi=fe + 1.96 * se,
                p=float(2 * (1 - st.norm.cdf(abs(z)))), weights=list(map(float, w / W)))


def pooled_tost(P, margin):
    """Equivalence of the pooled estimate within +-margin (two one-sided z-tests)."""
    p_lo = 1 - st.norm.cdf((P["est"] + margin) / P["se"])
    p_hi = st.norm.cdf((P["est"] - margin) / P["se"])
    return float(max(p_lo, p_hi))


def four_way(p_diff, p_tost, alpha):
    if p_diff < alpha and p_tost < alpha: return "TRIVIALLY DIFFERENT"
    if p_diff < alpha: return "DIFFERENT"
    if p_tost < alpha: return "EQUIVALENT"
    return "INCONCLUSIVE"


def contrast(C, block, game, mlp_arm, attn_arm, key="coord", gate="best"):
    x = values(C, block, game, mlp_arm, key, gate); y = values(C, block, game, attn_arm, key, gate)
    w = sst.welch(x, y)
    w.update(n_mlp=int(len(x)), n_attn=int(len(y)),
             n_registered=n_registered(block), mean_mlp=float(x.mean()), mean_attn=float(y.mean()))
    return w


# ------------------------------------------------------------------ hypotheses
def h1(C):
    """PRIMARY H1: pooled matched contrast (attn_cap - mlp_lr012), four games."""
    b = A["h1"]
    per = {g: contrast(C, b["block"], g, b["mlp_arm"], b["attn_arm"]) for g in b["games"]}
    P = pooled(list(per.values()))
    pt = pooled_tost(P, b["margin"])
    verdict = four_way(P["p"], pt, b["alpha"])
    if P["p"] < b["alpha"] and P["est"] < 0:
        decision = "SUPPORTED"          # attention selects a lower coordinate
    elif P["p"] < b["alpha"] and P["est"] > 0:
        decision = "CONTRADICTED (opposite sign)"
    elif pt < b["alpha"]:
        decision = "NOT SUPPORTED (equivalent within margin)"
    else:
        decision = "NOT SUPPORTED (inconclusive)"
    return dict(per_game=per, pooled=P, tost_p=pt, verdict=verdict, decision=decision,
                original_estimate=b["original_estimate"])


def h2(C):
    """PRIMARY H2: on kuhn_dupcall the attention arm's split of payoff-identical
    actions is below 0.5 on both split coordinates (one-sample t, Holm over 2)."""
    b = A["h2"]
    out = {}
    for k in b["coords"]:
        v = values(C, b["block"], b["game"], b["attn_arm"], key=k)
        t, p = st.ttest_1samp(v, 0.5)
        se = v.std(ddof=1) / np.sqrt(len(v)); c = st.t.ppf(.975, len(v) - 1)
        out[k] = dict(n=int(len(v)), mean=float(v.mean()), lo=float(v.mean() - c * se),
                      hi=float(v.mean() + c * se), p=float(p))
    adj = sst.holm([out[k]["p"] for k in b["coords"]])
    for k, a in zip(b["coords"], adj):
        out[k]["p_holm"] = float(a)
    ok = all(out[k]["p_holm"] < b["alpha"] and out[k]["mean"] < 0.5 for k in b["coords"])
    return dict(coords=out, decision="SUPPORTED" if ok else "NOT SUPPORTED")


def secondary(C):
    """Declared secondary family; Holm-adjusted together. Reported, never used to
    overturn the primary decisions."""
    S = {}
    b1 = A["h1"]
    # S1: Kuhn alone, matched @ lr .012
    S["S1_kuhn_matched_lr012"] = contrast(C, b1["block"], "kuhn", b1["mlp_arm"], b1["attn_arm"])
    # S2: pooled matched contrast with FINAL-iterate coordinate and final-iterate gate
    per_f = [contrast(C, b1["block"], g, b1["mlp_arm"], b1["attn_arm"], key="coord_final", gate="final")
             for g in b1["games"]]
    S["S2_pooled_final_iterate"] = pooled(per_f)
    # S3: Kuhn, least-squares projection metric instead of read-off
    S["S3_kuhn_projection"] = projection_contrast(C, b1["block"], b1["mlp_arm"], b1["attn_arm"])
    # S4: Kuhn, matched @ lr .015 (mlp vs attn_cap15)
    b4 = A["s4"]
    S["S4_kuhn_matched_lr015"] = contrast(C, b4["block"], "kuhn", b4["mlp_arm"], b4["attn_arm"])
    # S5: kuhn_dupcall, MLP split equivalent to 0.5 within +-margin (TOST, one-sample)
    S["S5_mlp_split_equivalence"] = mlp_split_equivalence(C)
    # S6: training moves attention's split (paired trained - untrained, same seeds)
    S["S6_split_trained_minus_untrained"] = split_vs_init(C)
    keys = ["S1_kuhn_matched_lr012", "S2_pooled_final_iterate", "S3_kuhn_projection",
            "S4_kuhn_matched_lr015"]
    ps = [S[k]["p"] for k in keys]
    for k, a in zip(keys, sst.holm(ps)):
        S[k]["p_holm"] = float(a)
    # descriptive: dependence-robust pooling of H1 (no test, a sensitivity curve)
    per = [contrast(C, b1["block"], g, b1["mlp_arm"], b1["attn_arm"]) for g in b1["games"]]
    S["equicorrelation_sweep"] = {str(r): pooled(per, rho=r) for r in (0.0, 0.3, 0.5, 0.7, 0.9)}
    return S


def projection_contrast(C, block, mlp_arm, attn_arm):
    g = efg.GAMES["kuhn"]; fam = efg.kuhn_family_strategy
    blk = next(b for b in CFG["blocks"] if b["name"] == block)
    vals = {}
    for arm in (mlp_arm, attn_arm):
        v = []
        for s in range(blk["seed_start"], blk["seed_start"] + blk["n_per_arm"]):
            r = C.get("%s|kuhn|%s|%d" % (block, arm, s))
            if r is None or r["exploit"] >= CONV or not r.get("numeric_ok", True):
                continue
            sig = {eval(k): np.array(x) for k, x in r["sigma"].items()}
            v.append(sst.project_coord(fam, 0.0, 1 / 3, sig, g)[0])
        vals[arm] = np.array(v)
    w = sst.welch(vals[mlp_arm], vals[attn_arm])
    w.update(n_mlp=int(len(vals[mlp_arm])), n_attn=int(len(vals[attn_arm])))
    return w


def mlp_split_equivalence(C):
    b = A["h2"]; m = b["mlp_split_margin"]; out = {}
    for k in b["coords"]:
        v = values(C, b["block"], b["game"], b["mlp_arm"], key=k)
        se = v.std(ddof=1) / np.sqrt(len(v)); df = len(v) - 1
        p_lo = 1 - st.t.cdf((v.mean() - 0.5 + m) / se, df)
        p_hi = st.t.cdf((v.mean() - 0.5 - m) / se, df)
        out[k] = dict(n=int(len(v)), mean=float(v.mean()), tost_p=float(max(p_lo, p_hi)))
    return out


def split_vs_init(C):
    """Paired: for each registered seed, trained split minus the split of the SAME
    network at initialization (nbn.init_params with that seed, as nbn.train does)."""
    b = A["h2"]; g = games2.GAMES2[b["game"]][0]; F = nbn.build_features(g)
    blk = next(x for x in CFG["blocks"] if x["name"] == b["block"])
    kw = {"mlp": dict(h=16, d=12), "attn": dict(h=16, d=8)}
    out = {}
    for arm in (b["mlp_arm"], b["attn_arm"]):
        diffs = {k: [] for k in b["coords"]}
        for s in range(blk["seed_start"], blk["seed_start"] + blk["n_per_arm"]):
            r = C.get("%s|%s|%s|%d" % (b["block"], b["game"], arm, s))
            if r is None or r["exploit"] >= CONV or not r.get("numeric_ok", True):
                continue
            p0 = nbn.init_params(arm, F, seed=s, **kw[arm])
            sig0 = nbn.rows_to_sigma(np.array(nbn.forward(arm, p0, F)), F)
            a0 = g.aux_coords(sig0)
            for k in b["coords"]:
                diffs[k].append(r[k] - a0[k])
        for k in b["coords"]:
            d = np.array(diffs[k]); t, p = st.ttest_1samp(d, 0.0)
            se = d.std(ddof=1) / np.sqrt(len(d)); c = st.t.ppf(.975, len(d) - 1)
            out["%s|%s" % (arm, k)] = dict(n=int(len(d)), mean=float(d.mean()),
                                          lo=float(d.mean() - c * se), hi=float(d.mean() + c * se),
                                          p=float(p))
    return out


# ------------------------------------------------------------------ report
def completeness(C):
    import stageE
    missing = [stageE.key(j) for j in stageE.jobs() if stageE.key(j) not in C]
    return dict(registered=len(stageE.jobs()), present=len(stageE.jobs()) - len(missing),
                missing=missing[:20])


def main(path):
    C = json.load(open(path))
    R = dict(completeness=completeness(C), H1=h1(C), H2=h2(C), secondary=secondary(C))
    H1, H2, S = R["H1"], R["H2"], R["secondary"]
    print("STAGE E REGISTERED ANALYSIS\n")
    print("runs present %d / %d registered" % (R["completeness"]["present"], R["completeness"]["registered"]))
    print("\nH1  matched contrast attn_cap - mlp_lr012 (read-off, best-iterate gate)")
    for g, w in H1["per_game"].items():
        print("   %-13s n=(%d,%d)/%d  %+.5f [%+.5f,%+.5f] p=%.4g"
              % (g, w["n_mlp"], w["n_attn"], w["n_registered"], w["diff"], w["lo"], w["hi"], w["p"]))
    P = H1["pooled"]
    print("   POOLED        %+.5f [%+.5f,%+.5f] p=%.4g  TOST(+-%.3f) p=%.4g"
          % (P["est"], P["lo"], P["hi"], P["p"], A["h1"]["margin"], H1["tost_p"]))
    print("   verdict: %s   decision: %s   (original post hoc estimate %+.5f)"
          % (H1["verdict"], H1["decision"], H1["original_estimate"]))
    print("\nH2  kuhn_dupcall attention split vs exact 0.5")
    for k, v in H2["coords"].items():
        print("   %-8s n=%d mean=%.4f [%.4f,%.4f] p=%.3g Holm p=%.3g"
              % (k, v["n"], v["mean"], v["lo"], v["hi"], v["p"], v["p_holm"]))
    print("   decision: %s" % H2["decision"])
    print("\nSecondary family (Holm across S1-S4)")
    for k in ("S1_kuhn_matched_lr012", "S2_pooled_final_iterate", "S3_kuhn_projection", "S4_kuhn_matched_lr015"):
        w = S[k]; est = w.get("diff", w.get("est"))
        print("   %-26s %+.5f [%+.5f,%+.5f] p=%.4g Holm p=%.4g" % (k, est, w["lo"], w["hi"], w["p"], w["p_holm"]))
    print("   S5 MLP split TOST(+-%.3f):" % A["h2"]["mlp_split_margin"],
          {k: round(v["tost_p"], 4) for k, v in S["S5_mlp_split_equivalence"].items()})
    print("   S6 trained - untrained (paired):")
    for k, v in S["S6_split_trained_minus_untrained"].items():
        print("      %-18s %+.4f [%+.4f,%+.4f] p=%.3g" % (k, v["mean"], v["lo"], v["hi"], v["p"]))
    print("   H1 pooled under equicorrelation rho:",
          {r: "%.3g" % v["p"] for r, v in S["equicorrelation_sweep"].items()})
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(path)), "stageE_results.json"), "w"),
              indent=1, default=float)
    return R


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stageE_cache.json")

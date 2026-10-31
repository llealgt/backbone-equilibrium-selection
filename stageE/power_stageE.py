"""A priori power for the Stage-E registration, computed from EXISTING (Stage C/D)
data only. Uses the observed per-game standard deviations; the effect sizes are
fixed smallest-effects-of-interest, not the observed estimates (which are
probably inflated by selection: they are the results that prompted Stage E).

    python power_stageE.py <dir with stageC_cache.json and stageD_cache.json>
"""
import json, os, sys
import numpy as np
from scipy import stats as st

D = sys.argv[1] if len(sys.argv) > 1 else "."
SC = json.load(open(os.path.join(D, "stageC_cache.json")))
SD = json.load(open(os.path.join(D, "stageD_cache.json")))
CONV = 0.02
GAMES = ["kuhn", "kuhn_dupcall", "kuhn_dupp1", "kuhn_dupbet"]


def gated(C, prefix, n, key="coord"):
    v = []
    for s in range(n):
        r = C.get("%s|%d" % (prefix, s))
        if r and key in r and r["exploit"] < CONV and r.get("numeric_ok", True):
            v.append(r[key])
    return np.array(v)


# ---------------- H1: pooled matched contrast, equal n per game ----------------
SESOI_H1 = 0.004      # ~ instrument resolution (0.0047), about half the exploratory estimate
ALPHA_H1 = 0.025      # Bonferroni over the two primary hypotheses
sd = {}
for g in GAMES:
    n = 150 if g == "kuhn" else 40
    m = gated(SD, "%s|mlp_lr012" % g, n); a = gated(SD, "%s|attn_cap" % g, n)
    sd[g] = (m.std(ddof=1), a.std(ddof=1), len(m), len(a))
print("H1 per-game SDs from Stage D (mlp_lr012, attn_cap):")
for g, (sm, sa, nm, na) in sd.items():
    print("   %-13s sd_mlp=%.4f (n=%d)  sd_attn=%.4f (n=%d)" % (g, sm, nm, sa, na))


def h1_power(n, delta=SESOI_H1, alpha=ALPHA_H1, rho=0.0):
    se = np.array([np.sqrt(sm**2 / n + sa**2 / n) for sm, sa, _, _ in sd.values()])
    w = 1 / se**2; W = w.sum()
    cov = sum(w[i] * w[j] * rho * se[i] * se[j] for i in range(4) for j in range(4) if i != j)
    se_p = np.sqrt(((w**2 * se**2).sum() + cov) / W**2)
    z = st.norm.ppf(1 - alpha / 2)
    return float(st.norm.cdf(delta / se_p - z)), float(se_p)


print("\nH1 power at SESOI=%.3f, alpha=%.3f two-sided:" % (SESOI_H1, ALPHA_H1))
print("   n/arm/game  power(rho=0)  power(rho=0.5)  pooled SE")
for n in (100, 150, 200, 250, 300, 400):
    p0, se0 = h1_power(n); p5, _ = h1_power(n, rho=0.5)
    print("   %-11d %-13.3f %-15.3f %.5f" % (n, p0, p5, se0))

# ---------------- H2: attention split on kuhn_dupcall ----------------
SESOI_H2 = 0.02       # about half of the smaller exploratory deviation (0.037)
ALPHA_H2 = 0.025 / 2  # Bonferroni share, then Holm over two coordinates (worst case)
print("\nH2 SDs from Stage C (attn, kuhn_dupcall, n=80):")
sds = {}
for k in ("split_Q", "split_K"):
    v = gated(SC, "kuhn_dupcall|attn", 80, key=k); sds[k] = v.std(ddof=1)
    print("   %-8s mean=%.4f sd=%.4f n=%d" % (k, v.mean(), sds[k], len(v)))
print("\nH2 power per coordinate at SESOI=%.3f, alpha=%.4f two-sided:" % (SESOI_H2, ALPHA_H2))
for n in (80, 100, 150, 200):
    pw = [st.nct.sf(st.t.ppf(1 - ALPHA_H2 / 2, n - 1), n - 1, SESOI_H2 / (s / np.sqrt(n))) for s in sds.values()]
    print("   n=%-4d " % n + "  ".join("%s %.3f" % (k, p) for k, p in zip(sds, pw)))

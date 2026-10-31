"""
selstats.py — inference utilities for equilibrium-selection comparisons.

Replaces the ad-hoc `effect > 2 * seed_sd` rule used in earlier versions of this
study. That rule compared a difference in means against a standard DEVIATION
rather than a standard ERROR, inflating the acceptance threshold by ~sqrt(n) and
biasing every verdict toward "invariant".

Everything here is deliberately explicit: point estimate, interval, test, and a
separate EQUIVALENCE test, because "we failed to reject a difference" and "we
demonstrated the difference is small" are different claims and only the second
one licenses the word invariant.
"""
import numpy as np
from scipy import stats

# ---------------------------------------------------------------- effect sizes
def hedges_g(x, y):
    """Bias-corrected standardised mean difference (Hedges' g)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    d = (np.mean(y) - np.mean(x)) / sp
    J = 1.0 - 3.0 / (4 * (nx + ny) - 9)          # small-sample correction
    return float(d * J)


# ---------------------------------------------------------------- interval estimation
def welch(x, y):
    """Welch two-sample comparison. Returns diff (y-x), SE, df, t, p, 95% CI."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    diff = float(np.mean(y) - np.mean(x))
    se = float(np.sqrt(vx / nx + vy / ny))
    if se == 0:
        return dict(diff=diff, se=0.0, df=np.nan, t=np.nan, p=np.nan, lo=diff, hi=diff)
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    t = diff / se
    p = float(2 * stats.t.sf(abs(t), df))
    crit = stats.t.ppf(0.975, df)
    return dict(diff=diff, se=se, df=float(df), t=float(t), p=p,
                lo=diff - crit * se, hi=diff + crit * se)


def boot_ci(x, y, B=20000, seed=0):
    """Bootstrap percentile CI on the difference of means — no normality assumption.

    Reported alongside Welch because seed-level selection coordinates are bounded
    (the Nash family has a finite coordinate range) and can be skewed near the
    boundary, which is exactly where the t-interval is least trustworthy.
    """
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, (B, len(x)), replace=True).mean(1)
    by = rng.choice(y, (B, len(y)), replace=True).mean(1)
    d = by - bx
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- equivalence
def tost(x, y, margin):
    """Two one-sided tests for equivalence within +-margin.

    p < 0.05 licenses the positive claim "the difference is smaller than margin".
    A large p means NOT-demonstrated-equivalent, which is NOT the same as
    "there is a difference".
    """
    w = dict(welch(x, y))
    w["p_welch"] = w.pop("p")
    if w["se"] == 0:
        w.update(p=0.0, margin=margin)
        return w
    p_lo = float(stats.t.sf((w["diff"] + margin) / w["se"], w["df"]))
    p_hi = float(stats.t.cdf((w["diff"] - margin) / w["se"], w["df"]))
    w.update(p=float(max(p_lo, p_hi)), margin=margin)
    return w


def tost_power(n_per_arm, sd, margin, true_diff=0.0, alpha=0.05, B=4000, seed=0):
    """Simulated power of the TOST at a given per-arm n. Used to size the study
    BEFORE running it, rather than reporting post-hoc power at the observed
    effect (which is a well-known statistical anti-pattern)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sd, (B, n_per_arm))
    y = rng.normal(true_diff, sd, (B, n_per_arm))
    diff = y.mean(1) - x.mean(1)
    se = np.sqrt(x.var(1, ddof=1) / n_per_arm + y.var(1, ddof=1) / n_per_arm)
    df = 2 * n_per_arm - 2
    p_lo = stats.t.sf((diff + margin) / se, df)
    p_hi = stats.t.cdf((diff - margin) / se, df)
    return float((np.maximum(p_lo, p_hi) < alpha).mean())


# ---------------------------------------------------------------- multiplicity
def holm(pvals):
    """Holm-Bonferroni adjusted p-values (step-down; uniformly more powerful
    than plain Bonferroni and still controls FWER under arbitrary dependence)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------- verdicts
def verdict(x, y, margin, alpha=0.05):
    """Four-way verdict. The old code had two outcomes; there are really four,
    and collapsing them is what produced the overclaim."""
    if len(x) < 2 or len(y) < 2:
        return "UNDER-POWERED"
    w = welch(x, y)
    e = tost(x, y, margin)
    diff_sig = w["p"] < alpha
    equiv = e["p"] < alpha
    if diff_sig and not equiv:
        return "DIFFERENT"
    if equiv and not diff_sig:
        return "EQUIVALENT (within +-%.3f)" % margin
    if equiv and diff_sig:
        return "TRIVIALLY DIFFERENT (real but < %.3f)" % margin
    return "INCONCLUSIVE (neither difference nor equivalence established)"


# ---------------------------------------------------------------- coordinate projection
def project_coord(fam, lo, hi, sigma, game, n=2001):
    """Least-squares projection of sigma onto the analytic Nash family.

    The earlier metric read a single component of sigma (e.g. P(bet|Jack)) and
    called it the selected coordinate. That is a read-off, not a projection: it
    ignores the other family constraints, so an off-manifold policy is assigned a
    coordinate it does not actually sit at. This returns argmin_c ||sigma - fam(c)||
    over the shared infosets, plus the residual distance off the family.
    """
    cs = np.linspace(lo, hi, n)
    keys = [I for I in sigma if I in fam(cs[0])]
    best = (np.inf, None)
    for c in cs:
        f = fam(c)
        d = np.sqrt(sum(float(np.sum((np.asarray(sigma[I]) - np.asarray(f[I])) ** 2)) for I in keys))
        if d < best[0]:
            best = (d, c)
    return float(best[1]), float(best[0])

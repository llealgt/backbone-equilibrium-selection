"""Draws the README figures from the result files in this repository.

    python make_readme_figures.py        # writes figs/readme/*_light.png and *_dark.png

Every number plotted is computed here from runcache.json, stage{B,C,D}_cache.json,
stageE/stageE_cache.json, stageE/stageE_results.json and recomputed_local/.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stageE"))
import selstats as sst                                   # the registered statistics module

OUT = os.path.join(HERE, "figs", "readme")
os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(os.path.join(HERE, p)))
CONV = 0.02

# ------------------------------------------------------------------ themes (validated palette)
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", band="#efeeea",
                  mlp="#2a78d6", attn="#eb6834", muted="#9a9994"),
    "dark":  dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#33332f", band="#2a2a27",
                  mlp="#3987e5", attn="#d95926", muted="#77766f"),
}


def style(T):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 160,
        "axes.facecolor": T["surface"], "figure.facecolor": T["surface"], "savefig.facecolor": T["surface"],
        "axes.edgecolor": T["ink2"], "axes.labelcolor": T["ink2"], "text.color": T["ink"],
        "xtick.color": T["ink2"], "ytick.color": T["ink2"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
    })


def save(fig, name, mode):
    fig.savefig(os.path.join(OUT, "%s_%s.png" % (name, mode)))
    plt.close(fig)


# ------------------------------------------------------------------ data
SD, SC, SE = J("stageD_cache.json"), J("stageC_cache.json"), J("stageE/stageE_cache.json")
RE = J("stageE/stageE_results.json")
GAMES = ["kuhn", "kuhn_dupcall", "kuhn_dupp1", "kuhn_dupbet"]
N_D = {"kuhn": 150, "kuhn_dupcall": 40, "kuhn_dupp1": 40, "kuhn_dupbet": 40}


def gated(C, prefix, seeds, key="coord"):
    out = []
    for s in seeds:
        r = C["%s|%d" % (prefix, s)]
        if r["exploit"] < CONV and r.get("numeric_ok", True):
            out.append(r[key])
    return np.array(out)


def pool(ws):
    y = np.array([w["diff"] for w in ws]); se = np.array([w["se"] for w in ws]); W = 1 / se**2
    m = (W * y).sum() / W.sum(); s = np.sqrt(1 / W.sum())
    return dict(diff=m, lo=m - 1.96 * s, hi=m + 1.96 * s)


explD = {g: sst.welch(gated(SD, "%s|mlp_lr012" % g, range(N_D[g])),
                      gated(SD, "%s|attn_cap" % g, range(N_D[g]))) for g in GAMES}
explD["pooled"] = pool([explD[g] for g in GAMES])
regE = dict(RE["H1"]["per_game"]); regE["pooled"] = dict(diff=RE["H1"]["pooled"]["est"],
                                                         lo=RE["H1"]["pooled"]["lo"], hi=RE["H1"]["pooled"]["hi"])

# H2: per-run splits (registered) and exploratory means with CIs (Stage C, n = 80)
def ci(v):
    from scipy import stats as st
    se = v.std(ddof=1) / np.sqrt(len(v)); c = st.t.ppf(.975, len(v) - 1)
    return v.mean(), v.mean() - c * se, v.mean() + c * se

H2 = {}
for arm in ("mlp", "attn"):
    for k in ("split_Q", "split_K"):
        H2[(arm, k, "E")] = gated(SE, "H2|kuhn_dupcall|%s" % arm, range(60000, 60400), key=k)
        H2[(arm, k, "C")] = gated(SC, "kuhn_dupcall|%s" % arm, range(80), key=k)

# reproducibility: original Colab files vs local recomputation
REPRO = []
for label, f in (("Stage A", "runcache.json"), ("Stage B", "stageB_cache.json"),
                 ("Stage C", "stageC_cache.json"), ("Stage D", "stageD_cache.json")):
    O, L = J(f), J(os.path.join("recomputed_local", f))
    same = sum(json.dumps(O[k], sort_keys=True) == json.dumps(L.get(k), sort_keys=True) for k in O)
    REPRO.append((label, same, len(O) - same))


# ------------------------------------------------------------------ figure 1: overview
def fig_overview(T, mode):
    style(T)
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 2.9); ax.axis("off")
    stages = [("A", "first study,", "small n", "130"),
              ("B", "Kuhn, n = 150/arm,", "LayerNorm/PE factorial", "800"),
              ("C", "3 constructed games,", "exact Nash families", "840"),
              ("D", "size- and step-size-", "matched arms", "690"),
              ("E", "fresh-seed replication", "of the C and D findings", "4,800")]
    x0, w, gap = 0.15, 2.0, 0.2
    for i, (s, l1, l2, n) in enumerate(stages):
        x = x0 + i * (w + gap) + (0.25 if i == 4 else 0)
        reg = (s == "E")
        box = FancyBboxPatch((x, 0.75), w, 1.35, boxstyle="round,pad=0,rounding_size=0.08",
                             linewidth=1.6 if reg else 1.0,
                             edgecolor=T["mlp"] if reg else T["muted"],
                             facecolor=T["mlp"] if reg else T["surface"])
        ax.add_patch(box)
        tc = "#ffffff" if reg else T["ink"]; tc2 = "#ffffff" if reg else T["ink2"]
        ax.text(x + 0.14, 1.86, "Stage " + s, fontsize=11, weight="bold", color=tc, va="center")
        ax.text(x + 0.14, 1.50, l1, fontsize=8.6, color=tc2, va="center")
        ax.text(x + 0.14, 1.27, l2, fontsize=8.6, color=tc2, va="center")
        ax.text(x + 0.14, 0.95, n + " runs", fontsize=9, color=tc, va="center", weight="bold")
    # exploratory bracket
    xe = x0 + 4 * (w + gap) - gap
    ax.plot([x0, xe], [0.5, 0.5], color=T["muted"], lw=1.2)
    ax.text((x0 + xe) / 2, 0.28, "exploratory: each stage designed after seeing the previous one",
            ha="center", fontsize=9, color=T["ink2"])
    xr = x0 + 4 * (w + gap) + 0.25
    ax.plot([xr, xr + w], [0.5, 0.5], color=T["mlp"], lw=1.6)
    ax.text(xr + w / 2, 0.28, "confirmatory", ha="center", fontsize=9, color=T["ink"], weight="bold")
    # registration marker
    xm = xr - 0.2
    ax.plot([xm, xm], [0.6, 2.35], color=T["ink"], lw=1.2, ls=(0, (3, 2)))
    ax.text(xm, 2.5, "registered on OSF before any Stage-E run\nosf.io/dgyuq  ·  2026-10-31 19:07 UTC",
            ha="center", fontsize=8.6, color=T["ink"], va="bottom")
    save(fig, "overview", mode)


# ------------------------------------------------------------------ figure 2: H1 forest
def fig_h1(T, mode):
    style(T)
    rows = GAMES + ["pooled"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.axvspan(-0.010, 0.010, color=T["band"], lw=0, zorder=0)
    ax.axvline(0, color=T["ink2"], lw=0.9, zorder=1)
    for i, g in enumerate(rows):
        y = len(rows) - 1 - i + (0 if g != "pooled" else -0.35)
        e, r = explD[g], regE[g]
        ax.plot([e["lo"], e["hi"]], [y + 0.17] * 2, color=T["muted"], lw=2, solid_capstyle="round", zorder=2)
        ax.plot(e["diff"], y + 0.17, "o", ms=8, mfc=T["surface"], mec=T["muted"], mew=1.8, zorder=3)
        ax.plot([r["lo"], r["hi"]], [y - 0.17] * 2, color=T["mlp"], lw=2.6 if g == "pooled" else 2,
                solid_capstyle="round", zorder=2)
        ax.plot(r["diff"], y - 0.17, "D" if g == "pooled" else "o", ms=9 if g == "pooled" else 8,
                color=T["mlp"], mec=T["surface"], mew=1.5, zorder=4)
    labels = GAMES + ["pooled"]
    ax.set_yticks([len(rows) - 1 - i + (0 if g != "pooled" else -0.35) for i, g in enumerate(rows)])
    ax.set_yticklabels(labels)
    for t in ax.get_yticklabels():
        if t.get_text() == "pooled": t.set_fontweight("bold"); t.set_color(T["ink"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.022, 0.0125); ax.set_ylim(-1.0, len(rows) - 0.55)
    ax.set_xlabel("selected-coordinate difference, attention − MLP (95% CI)")
    ax.grid(axis="x", color=T["grid"], lw=0.7, zorder=0); ax.set_axisbelow(True)
    pr = regE["pooled"]
    ax.text(pr["lo"] - 0.0006, -0.52, "registered %+.4f" % pr["diff"], fontsize=8.8, color=T["ink"],
            weight="bold", va="center", ha="right")
    pe = explD["pooled"]
    ax.text(pe["lo"] - 0.0006, -0.18, "exploratory %+.4f" % pe["diff"], fontsize=8.8, color=T["ink2"],
            va="center", ha="right")
    ax.text(0.0097, -0.95, "shaded: ±0.010 margin set\nas practically relevant",
            fontsize=8.2, color=T["ink2"], ha="right", va="bottom")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=T["muted"], marker="o", mfc=T["surface"], mec=T["muted"], lw=2, ms=7,
                label="exploratory (Stage D, n = 150 / 40 per arm)"),
         Line2D([], [], color=T["mlp"], marker="o", lw=2, ms=7,
                label="registered replication (Stage E, n = 400 per arm)")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.02, 1.13), ncol=2, fontsize=8.6,
              handlelength=2.2, columnspacing=1.6)
    ax.set_title("H1: with size and learning rate matched, attention selects a lower equilibrium",
                 loc="left", fontsize=11, color=T["ink"], pad=34)
    save(fig, "h1_replication", mode)


# ------------------------------------------------------------------ figure 3: H2 split calibration
def fig_h2(T, mode):
    style(T)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    for ax, k, title in zip(axes, ("split_Q", "split_K"), ("Queen information set", "King information set")):
        ax.axhline(0.5, color=T["ink"], lw=1.1, zorder=1)
        for j, (arm, c, name) in enumerate((("mlp", T["mlp"], "MLP"), ("attn", T["attn"], "attention"))):
            v = H2[(arm, k, "E")]
            x = j + rng.uniform(-0.22, 0.22, len(v))
            ax.scatter(x, v, s=7, color=c, alpha=0.35, lw=0, zorder=2)
            m, lo, hi = ci(v)
            me, loe, hie = ci(H2[(arm, k, "C")])
            ax.plot([j + 0.31, j + 0.31], [loe, hie], color=T["muted"], lw=1.8, solid_capstyle="round", zorder=3)
            ax.plot(j + 0.31, me, "o", ms=6, mfc=T["surface"], mec=T["muted"], mew=1.5, zorder=4)
            ax.plot([j + 0.43, j + 0.43], [lo, hi], color=c, lw=3, solid_capstyle="butt", zorder=3)
            ax.plot(j + 0.43, m, "_", ms=12, mew=2.5, color=c, zorder=4)
            ax.text(j + 0.55, m, "%.3f" % m, fontsize=8.6, color=T["ink"], va="center", weight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["MLP", "attention"])
        ax.set_xlim(-0.35, 1.85); ax.set_ylim(0.32, 0.62)
        ax.set_title(title, fontsize=9.5, color=T["ink2"], loc="left")
        ax.grid(axis="y", color=T["grid"], lw=0.7, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("share of call probability on the first of two identical actions")
    axes[0].text(-0.33, 0.604, "horizontal line: exact maximum-entropy split, 0.5", fontsize=8.2,
                 color=T["ink2"], ha="left", va="bottom")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=T["ink2"], alpha=0.45, ms=4, label="one registered run (400 per arm)"),
         Line2D([], [], color=T["ink2"], marker="_", lw=3, ms=10, mew=2.5, label="registered mean, 95% CI"),
         Line2D([], [], color=T["muted"], marker="o", mfc=T["surface"], mec=T["muted"], lw=1.8, ms=6,
                label="exploratory mean (Stage C, n = 80)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, 1.02), ncol=3, fontsize=8.4)
    fig.suptitle("H2: attention splits payoff-identical actions unevenly; the MLP does not",
                 x=0.06, ha="left", y=1.1, fontsize=11, color=T["ink"])
    save(fig, "h2_split", mode)


# ------------------------------------------------------------------ figure 4: reproducibility
def fig_repro(T, mode):
    style(T)
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    for i, (lab, same, diff) in enumerate(REPRO[::-1]):
        ax.barh(i, same, height=0.56, color=T["mlp"], edgecolor=T["surface"], linewidth=2)
        if diff:
            ax.barh(i, diff, left=same, height=0.56, color=T["attn"], edgecolor=T["surface"], linewidth=2)
        if same > 300:
            ax.text(same / 2, i, "%d identical" % same, ha="center", va="center", fontsize=8.6,
                    color="#ffffff", weight="bold")
        else:
            ax.text(same + 10, i, "%d identical (all)" % same, ha="left", va="center", fontsize=8.6,
                    color=T["ink"])
        if diff:
            ax.text(same + diff + 8, i, "%d differ (LayerNorm arms)" % diff, ha="left", va="center",
                    fontsize=8.6, color=T["ink"])
    ax.set_yticks(range(len(REPRO))); ax.set_yticklabels([r[0] for r in REPRO[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1150); ax.set_xlabel("runs")
    ax.grid(axis="x", color=T["grid"], lw=0.7); ax.set_axisbelow(True)
    ax.set_title("Recomputed on another machine: every run without LayerNorm reproduces bit for bit",
                 loc="left", fontsize=11, color=T["ink"], pad=10)
    save(fig, "reproducibility", mode)


for mode, T in THEMES.items():
    fig_overview(T, mode); fig_h1(T, mode); fig_h2(T, mode); fig_repro(T, mode)
print("wrote", sorted(os.listdir(OUT)))
print("reproducibility:", REPRO)
print("exploratory pooled:", {k: round(v, 5) for k, v in explD["pooled"].items()})

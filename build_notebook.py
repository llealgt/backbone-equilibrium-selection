"""Builds backbone_study.ipynb from the historical notebook.

Code cells that produced the published numbers are kept; only wording about
"pre-registration" changes (Stages A-D are exploratory) and new sections are
added (Stage D metric checks, Stage E registered replication, reproducibility
check). The four modules shared with the Stage-E registration (efg, nbn, games2,
selstats) are copied byte-for-byte; the notebook verifies that at run time.

    python build_notebook.py
"""
import json, re, copy, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = next(p for p in (os.path.join(HERE, "history", "neural_selection_study_FINAL.ipynb"),
                       os.path.join(HERE, "..", "..", "neural_selection_study_FINAL.ipynb")) if os.path.exists(p))
OUT = os.path.join(HERE, "backbone_study.ipynb")

nb = json.load(open(SRC))
cells = nb["cells"]
REGISTERED_MODULE_CELLS = {6, 7, 8, 10}          # efg, nbn, selstats, games2


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.splitlines(True)


# ---------------------------------------------------------------- wording pass
WORDING = [
    ("PRE-REGISTRATION (FROZEN BEFORE ANY STAGE-B RUN)", "ANALYSIS PLAN (written during the study; NOT registered)"),
    ("PRE-REGISTRATION FROZEN", "ANALYSIS PLAN (historical; not registered, no external timestamp)"),
    ("Pre-registration extension", "Margins for the constructed games"),
    ("pre-registered primary endpoint", "Stage-B primary comparison"),
    ("PRIMARY ENDPOINT (pre-registered)", "STAGE-B PRIMARY COMPARISON (exploratory)"),
    ("pre-registered rule", "four-way rule"),
    ("pre-registered margin", "planned margin"),
    ("pre-registered metric", "planned metric"),
    ("pre-registered analysis", "planned analysis"),
    ("pre-registered (best-iterate) metric", "planned (best-iterate) metric"),
    ("Four-way, pre-registered.", "Four-way."),
    ("pre-registered", "planned"),
    ("Pre-registered", "Planned"),
    ("pre-registration", "analysis plan"),
    ("Pre-registration", "Analysis plan"),
    ("pre-specified", "planned"),
    ("the frozen plan", "the analysis plan"),
    ("frozen margin", "planned margin"),
    ("frozen before", "fixed before"),
    ("was frozen", "was set"),
    ("frozen", "planned"),
    ("Confirmatory results", "Results (exploratory)"),
    ("CONFIRMATORY", "STAGE-B"),
    ("""# This cell fixes the analysis plan BEFORE the confirmatory runs execute. Nothing
# below may be edited after Stage-B results exist; re-running it re-prints the
# same constants and re-computes the same fingerprint, so any post-hoc edit is
# visible in a diff. This exists because""",
     """# Historical record: this plan was written into the notebook while Stages B-D were
# being run. It was never registered and has no external timestamp, so it cannot
# show when it was written; everything it governs is exploratory. Later cells read
# MARGIN and the verdict rule from it. It exists because"""),
    ("Confirmatory runs", "Runs"),
    ("the confirmatory caches", "the result caches"),
    ("confirmatory data", "large-sample data"),
    ("confirmatory estimate", "large-sample estimate"),
    ("800 confirmatory runs", "800 Stage-B runs"),
    ("confirmatory driver", "Stage-B driver"),
    ("The confirmatory version of this analysis", "The large-sample version of this analysis"),
    ("the confirmatory pool and excluded from every\nconfirmatory analysis", "the analysed seeds and excluded from every\nanalysis"),
    ("per-game confirmatory n", "per-game n"),
    ("NOTHING in the paper rests on this: the confirmatory", "Only the audit uses this tier: the large-sample"),
    ("A confirmatory\")", "A large-sample\")"),
    ('"confirmatory  n=', '"Stage B  n='),
    ("the confirmatory Stage-B data", "the Stage-B data"),
    ("PREREG", "PLAN"),
]
for i, c in enumerate(cells):
    if i in REGISTERED_MODULE_CELLS:
        continue                                   # byte-identical to the registration
    s = src(i)
    for a, b in WORDING:
        s = s.replace(a, b)
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
    set_src(i, s)

# remove the local fingerprint print: it was never an external timestamp
s = src(17)
s = re.sub(r'_fp = hashlib\.sha256\(.*?\n', '', s)
s = re.sub(r'print\("  fingerprint \(sha256\[:16\]\) :", _fp\)\n', '', s)
s = s.replace('print("  branches : both interpretations pre-committed (see branch_1 / branch_2 above)")',
              'print("  note     : this plan has no external timestamp; Stages A-D are exploratory.")')
set_src(17, s)

# ---------------------------------------------------------------- rewritten markdown
set_src(0, """# Does the Backbone Choose the Equilibrium? — complete, reproducible notebook

This notebook regenerates **every number, table and figure** in the paper, including the registered
confirmatory replication (Stage E, https://osf.io/dgyuq).

| stage | what | runs | status | result file |
|---|---|---|---|---|
| A | first study, small seed counts (sections 1–12) | exploratory tier | exploratory | `runcache.json` |
| B | Kuhn: attention vs MLP at n=150, LayerNorm × PE factorial, sweeps | 800 | exploratory | `stageB_cache.json` |
| C | three constructed games with closed-form Nash families | 840 | exploratory | `stageC_cache.json` |
| D | capacity- and learning-rate-matched arms | 690 | exploratory | `stageD_cache.json` |
| E | fresh-seed replication of D's matched effect and C's split result | 4,800 | **registered** before any run | `stageE/stageE_cache.json` |

**Exploratory vs confirmatory.** Stages B–D were each designed after the previous stages' data had been
seen. An analysis plan was written in this notebook during that sequence (section 0b), but it was never
registered and carries no external timestamp, so Stages A–D are exploratory. Stage E's hypotheses, seeds,
sample sizes, tests and decision rules were registered on OSF before any Stage-E run; its code is the
registered zip, verified by SHA-256 in section E0.

**How to run.** Run all cells top to bottom. Every stage loads its result file if it is present and
recomputes it from scratch if it is not, so you can either check the analyses quickly (shipped results)
or regenerate everything (delete the `*.json` result files first). Recomputation is deterministic given
the package versions (Python 3.13, NumPy 2.1.3, SciPy 1.16.3, autograd 1.9.1).
Approximate cost on 18 CPU cores: Stage A ~1 h (single process), B 11 min, C 10 min, D 8 min, E 40 min.
On a 2-core Colab runtime, multiply the parallel stages by about 9.

**Where things are.** The source modules are written to disk by the cells in "Source modules". Stage E
runs from `stageE/`, unpacked from the registered zip. Figures go to `figs/`.
""")

set_src(16, """## 0b · Analysis plan written during the study (not registered)

The first version of this study decided equilibrium-selection questions with the rule
`effect > 2 * seed_sd`, which compares a difference in means against a standard **deviation**
instead of a standard **error**, so its threshold never tightens with sample size and every verdict
is biased toward "invariant".

The cell below records the corrected analysis plan that was written while Stages B–D were being run:
one primary comparison, an equivalence margin with its derivation, secondary analyses and a four-way
verdict rule. It is kept as a historical record and because later cells read the margin and threshold
from it. **It was not registered and has no external timestamp, so it cannot show when it was written;
everything it governs is exploratory.** The registered plan is Stage E (https://osf.io/dgyuq).
""")

set_src(61, """---

# Stage B · Results (exploratory)

Everything above this line is the small-sample exploratory tier (Stage A). Stage B is 800 training runs
on Kuhn: a 150-seed-per-arm comparison of attention against the MLP, a six-arm LayerNorm × positional-
encoding factorial, a four-point magnet sweep and a five-point training-length sweep. Stage B was
designed after Stage A, so it is exploratory as well.

Results load from `stageB_cache.json`. Delete that file and re-run this section to regenerate all 800
runs (`stageB.py`).
""")

set_src(63, """## B1 · The Stage-B primary comparison

Attention vs MLP on the selected equilibrium coordinate, Kuhn, n = 150 per arm, convergence-gated, TOST
at the planned margin of ±0.010.

The exploratory tier above ran this comparison at small n and obtained a **positive** estimate. At
n = 150 the estimate is **negative**. The sign flip is the clearest demonstration of why the retired
`effect > 2*sd` rule was dangerous: it accepted a null on an estimate whose direction was not yet
determined.
""")

set_src(77, """## C1 · Margins for the constructed games

Each constructed game gets its own equivalence margin from one procedure, applied to its own curvature
and shortfall:

> margin = the coordinate whose entropy cost equals 25% of that game's intrinsic tabular shortfall,
> $m=\\sqrt{2\\cdot 0.25\\cdot\\delta_{tab}/\\kappa}$, floored at $2\\times$ the table-limit resolution.

Applied to Kuhn this returns $0.01020 \\to 0.010$, the Kuhn margin. The procedure was written after
Stage B, so it is part of the exploratory record.

Sample sizes come from simulated TOST power on a pilot run at seeds 900–911, disjoint from the analysed
seeds.

**Questions these games enable:**
- **S1 duplicate-split calibration.** The max-entropy split of payoff-identical actions is exactly 0.5.
- **S2 reach ordering.** Is the effect largest where free parameters sit at high reach (`kuhn_dupbet`)?
- **S3 own-side vs opponent-side** (`kuhn_dupcall` vs `kuhn_dupp1`).
- **S4 cross-game meta-analysis** with heterogeneity, plus Holm across games.
""")

# ---------------------------------------------------------------- workspace cell (rewritten)
set_src(2, """# ---- workspace and result files ----------------------------------------------------
# Local run: everything lives in the current directory.
# Colab: files live in /content/backbone_study (or in Google Drive if USE_DRIVE = True,
#        so a disconnected runtime keeps its results).
# Missing result files are downloaded from the GitHub repository unless RECOMPUTE = True,
# in which case every missing stage is recomputed from scratch by the cells below.
import os, json, urllib.request
RECOMPUTE = False
USE_DRIVE = False
REPO_RAW  = "https://raw.githubusercontent.com/llealgt/backbone-equilibrium-selection/main/"

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False
if IN_COLAB:
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        WORKDIR = "/content/drive/MyDrive/backbone_study"
    else:
        WORKDIR = "/content/backbone_study"
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)
else:
    WORKDIR = os.getcwd()

RESULT_FILES = {
    "runcache.json":              "Stage A, exploratory tier (130 runs)",
    "stageB_cache.json":          "Stage B (800 runs)",
    "stageC_cache.json":          "Stage C (840 runs)",
    "stageD_cache.json":          "Stage D (690 runs)",
    "stageE/stageE_cache.json":   "Stage E, registered (4,800 runs)",
}
for _f in RESULT_FILES:
    if not os.path.exists(_f) and not RECOMPUTE:
        os.makedirs(os.path.dirname(_f) or ".", exist_ok=True)
        try:
            urllib.request.urlretrieve(REPO_RAW + _f, _f)
            print("downloaded", _f)
        except Exception as _e:
            print("could not download %s (%s): that stage will be recomputed" % (_f, _e))
print("working directory:", WORKDIR)
for _f, _d in RESULT_FILES.items():
    _n = len(json.load(open(_f))) if os.path.exists(_f) else 0
    print("   %-26s %6s   %s" % (_f, _n if _n else "MISSING -> recompute", _d))
""")

# ---------------------------------------------------------------- pinned versions (new cell before the bootstrap)
PIN_CELL = code("""# ---- exact package versions --------------------------------------------------------
# All results were produced with these versions. Training is deterministic given the seed
# and the versions; with other versions, runs that use LayerNorm can differ in the 3rd-4th
# decimal (see the reproducibility note at the end).
import sys, subprocess, importlib.metadata as _md
PINS = {"numpy": "2.1.3", "scipy": "1.16.3", "autograd": "1.9.1"}
def _ver(p):
    try:
        return _md.version(p)
    except Exception:
        return None
_need = ["%s==%s" % (p, v) for p, v in PINS.items() if _ver(p) != v]
if _need:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *_need], check=True)
    raise RuntimeError("Installed %s. Restart the runtime (Colab: Runtime > Restart session), "
                       "then run all cells again." % _need)
print("package versions match:", PINS, "| python", sys.version.split()[0])""")

# ---------------------------------------------------------------- new cells
STAGE_D_EXTRA = [
    md("""### D5 · The matched contrast under the other two metrics

The matched contrast above uses the read-off coordinate at the best iterate. Here it is recomputed with
the least-squares projection (Kuhn) and with the final-iterate coordinate and gate (pooled over four
games). These are the numbers quoted in the paper's measurement audit."""),
    code("""if SD:
    _g = efg.GAMES["kuhn"]; _P = {}
    for _arm in ("mlp_lr012", "attn_cap"):
        _v = []
        for _s in range(150):
            _r = SD["kuhn|%s|%d" % (_arm, _s)]
            if _r["exploit"] < CONV and _r["numeric_ok"]:
                _sig = {eval(k): np.array(v) for k, v in _r["sigma"].items()}
                _v.append(sst.project_coord(efg.kuhn_family_strategy, 0.0, 1/3, _sig, _g)[0])
        _P[_arm] = np.array(_v)
    _w = sst.welch(_P["mlp_lr012"], _P["attn_cap"])
    print("Kuhn, projection metric      %+.5f [%+.5f,%+.5f] p=%.4f" % (_w["diff"], _w["lo"], _w["hi"], _w["p"]))
    _per = []
    for _gn, _n in (("kuhn",150),("kuhn_dupcall",40),("kuhn_dupp1",40),("kuhn_dupbet",40)):
        def _fv(arm):
            return np.array([SD["%s|%s|%d"%(_gn,arm,s)]["coord_final"] for s in range(_n)
                             if SD["%s|%s|%d"%(_gn,arm,s)]["final_exploit"] < CONV
                             and SD["%s|%s|%d"%(_gn,arm,s)]["numeric_ok"]])
        _a, _b = _fv("mlp_lr012"), _fv("attn_cap"); _per.append(sst.welch(_a, _b))
        print("  %-13s final-iterate n=(%d,%d) %+.5f p=%.4f" % (_gn, len(_a), len(_b), _per[-1]["diff"], _per[-1]["p"]))
    _y = np.array([w["diff"] for w in _per]); _se = np.array([w["se"] for w in _per]); _W = 1/_se**2
    _fe = (_W*_y).sum()/_W.sum(); _s = np.sqrt(1/_W.sum())
    print("Pooled, final-iterate        %+.5f [%+.5f,%+.5f] p=%.2g"
          % (_fe, _fe-1.96*_s, _fe+1.96*_s, 2*(1-_st.norm.cdf(abs(_fe/_s)))))"""),
]

STAGE_E = [
    md("""---

# Stage E · Registered confirmatory replication (https://osf.io/dgyuq)

Registered on OSF on 2026-10-31 at 19:07 UTC, before any Stage-E run. The registration fixes two primary
hypotheses, the seeds, the sample size (400 runs per arm per game, 4,800 runs), the tests, α = 0.025 per
hypothesis and the decision rules, and attaches the code as `stageE_registered_code.zip`
(SHA-256 `25ec7c72…3e52`).

- **H1:** pooled matched contrast `attn_cap − mlp_lr012` over the four games is negative.
- **H2:** on `kuhn_dupcall`, attention's split of payoff-identical actions is below 0.5 on both coordinates.

**E0** obtains the registered zip (local copy, or downloaded from OSF), checks its SHA-256 against the
registration, and checks that the modules written by this notebook are byte-identical to the registered
ones. **E1** runs the registered jobs if their results are missing. **E2** runs the registered analysis
script, unchanged. Its output is what the paper reports."""),
    code("""# ---- E0 · registered code: fetch, verify, unpack ------------------------------------
import hashlib, zipfile, urllib.request, subprocess, sys
REG_ZIP    = "stageE_registered_code.zip"
REG_SHA256 = "25ec7c72509fbbc4650033c6270370bd7c7c228ec5c31ea0ca490c6394e33e52"
REG_URL    = "https://osf.io/download/6ae63c639644805e411d30d4/"     # file attached to osf.io/dgyuq
if not os.path.exists(REG_ZIP):
    print("downloading the registered code from OSF ...")
    urllib.request.urlretrieve(REG_URL, REG_ZIP)
_h = hashlib.sha256(open(REG_ZIP, "rb").read()).hexdigest()
assert _h == REG_SHA256, "registered zip does not match the registration: %s" % _h
print("registered zip SHA-256 matches osf.io/dgyuq:", _h)
os.makedirs("stageE", exist_ok=True)
with zipfile.ZipFile(REG_ZIP) as _z:
    for _m in _z.namelist():
        _name = os.path.basename(_m)
        if _name and not os.path.exists(os.path.join("stageE", _name)):
            open(os.path.join("stageE", _name), "wb").write(_z.read(_m))
_sums = dict(l.split()[::-1] for l in open("stageE/SHA256SUMS.txt").read().strip().splitlines())
for _f, _d in sorted(_sums.items()):
    _got = hashlib.sha256(open(os.path.join("stageE", _f), "rb").read()).hexdigest()
    assert _got == _d, "stageE/%s differs from the registered file" % _f
print("all %d registered files verified in stageE/" % len(_sums))
for _f in ("efg.py", "nbn.py", "games2.py", "selstats.py"):
    _same = open(_f, "rb").read() == open(os.path.join("stageE", _f), "rb").read()
    print("  %-12s written by this notebook == registered copy: %s" % (_f, _same))
    assert _same, "the notebook's %s differs from the registered module" % _f"""),
    code("""# ---- E1 · registered runs: load, or run them exactly as registered ---------------------
_E_CACHE = os.path.join("stageE", "stageE_cache.json")
if not os.path.exists(_E_CACHE):
    _np = max(1, (os.cpu_count() or 2) - 2)
    print("Stage-E cache missing: running the 4,800 registered jobs on %d processes (deterministic) ..." % _np)
    _env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "stageE.py", "parallel", str(_np)], cwd="stageE", env=_env, check=True)
SE = json.load(open(_E_CACHE))
print("Stage-E cache: %d runs (4800 registered)" % len(SE))"""),
    code("""# ---- E2 · the registered analysis, run unchanged ---------------------------------------
_r = subprocess.run([sys.executable, "analyzeE.py", "stageE_cache.json"], cwd="stageE",
                    capture_output=True, text=True, check=True)
print(_r.stdout)
open(os.path.join("stageE", "stageE_report.txt"), "w").write(_r.stdout)"""),
]

REPRO = [
    md("""## Reproducibility check

Recomputes headline numbers from the result files and compares them with the paper. With the shipped
result files every check must pass exactly.

**Note on recomputation.** If the result files are deleted and Stages A-D are recomputed, every run that
does not use LayerNorm is reproduced bit for bit (verified: 1,980 of 2,460 runs). The 480 runs of the
LayerNorm arms (`mlp+ln`, `attn+ln`, `attn+ln+pe`) are sensitive to the CPU and BLAS build: LayerNorm's
mean and variance reductions round differently, and training amplifies that into the 3rd-4th decimal of
those arms' results (for example `attn+ln` vs MLP on Kuhn: -0.0106, p=0.001 in the shipped files from
Google Colab; -0.0100, p=0.003 recomputed on a local Linux machine). No conclusion changes. Stage E has no
LayerNorm arm."""),
    code("""_chk = []
def _expect(label, got, want, tol=5e-5):
    ok = abs(got - want) <= tol; _chk.append(ok)
    print("%-52s got %+.5f  paper %+.5f  %s" % (label, got, want, "OK" if ok else "MISMATCH"))
_expect("Stage B: MLP mean coordinate (n=150)", sb_gated("sel","mlp").mean(), 0.1862)
_expect("Stage B: attention mean coordinate (n=150)", sb_gated("sel","attn").mean(), 0.1802)
_expect("Stage B: attn+ln minus MLP (LayerNorm arm)", sb_gated("sel","attn+ln").mean() - sb_gated("sel","mlp").mean(), -0.0106, tol=6e-5)
_expect("Stage C: attention split_Q, kuhn_dupcall", sc_gated("kuhn_dupcall","attn",key="split_Q").mean(), 0.4626)
_expect("Stage C: attention split_K, kuhn_dupcall", sc_gated("kuhn_dupcall","attn",key="split_K").mean(), 0.4486)
_expect("Stage D: matched contrast, Kuhn", sst.welch(sd_gated("kuhn","mlp_lr012",150), sd_gated("kuhn","attn_cap",150))["diff"], -0.00836)
_R = json.load(open(os.path.join("stageE", "stageE_results.json")))
_expect("Stage E (registered): H1 pooled matched contrast", _R["H1"]["pooled"]["est"], -0.00760)
_expect("Stage E (registered): H2 attention split_Q", _R["H2"]["coords"]["split_Q"]["mean"], 0.4722)
_expect("Stage E (registered): H2 attention split_K", _R["H2"]["coords"]["split_K"]["mean"], 0.4632)
print("H1 decision:", _R["H1"]["decision"], "| H2 decision:", _R["H2"]["decision"])
print("\\nall checks pass" if all(_chk) else "\\nSOME CHECKS FAILED")"""),
]

# insert D5 after the Stage-D matched-contrast-on-games cell (103), Stage E after it
order = []
for i, c in enumerate(cells):
    if i == 2:
        order.append(PIN_CELL)
    order.append(c)
    if i == 103:
        order.extend(STAGE_D_EXTRA)
        order.extend(STAGE_E)
order.extend(REPRO)
for _k, _c in enumerate(order):
    _c["id"] = "cell-%03d" % _k
nb["cells"] = order
nb["nbformat"], nb["nbformat_minor"] = 4, 5
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
for k in ("colab", "accelerator"):
    nb["metadata"].pop(k, None)
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT, "with", len(order), "cells")

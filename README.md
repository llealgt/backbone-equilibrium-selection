# Does the Backbone Choose the Equilibrium?

Code, data and the registered replication for the paper *Does the Backbone Choose the Equilibrium? A
Controlled Multi-Game Study of Architecture and Measurement in Regularized Self-Play* (L. Leal, 2026;
arXiv link to be added).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/llealgt/backbone-equilibrium-selection/blob/main/backbone_study.ipynb)

**One notebook, `backbone_study.ipynb`, reproduces every number, table and figure in the paper.**

## The question

Regularized self-play (R-NaD, the algorithm behind DeepNash) converges to one particular member of a
game's set of Nash equilibria. Does the policy network's architecture change which member? Counterfactual
values are computed exactly, the solver is fixed, and only the network varies, on Kuhn poker and three
constructed variants whose Nash families are known in closed form.

## Results in brief

| stage | runs | status | main result |
|---|---|---|---|
| A | 130 | exploratory | first study; its "invariant" verdict came from a flawed decision rule |
| B | 800 | exploratory | Kuhn: attention vs MLP, LayerNorm × positional-encoding factorial, sweeps |
| C | 840 | exploratory | three constructed games; attention miscalibrates a split of payoff-identical actions |
| D | 690 | exploratory | parameter count and learning rate matched: pooled effect −0.0083 |
| **E** | **4,800** | **registered before any run: [osf.io/dgyuq](https://osf.io/dgyuq)** | **both findings replicate** |

Stage E (registered hypotheses, 400 runs per arm per game, new seeds):

- **H1, supported.** Matched contrast (attention − MLP) pooled over four games: **−0.0076**,
  95% CI [−0.0090, −0.0062], significant in every game separately. It is also significantly smaller than
  the ±0.010 margin we set as practically relevant, so the effect is real but small.
- **H2, supported.** On `kuhn_dupcall` attention splits two payoff-identical actions at **0.472** and
  **0.463** against the exact 0.5; the MLP stays at 0.497 and 0.498. Training causes the shift.

Full output of the registered analysis: [`stageE/stageE_report.txt`](stageE/stageE_report.txt).

## Running the notebook

**In Colab:** click the badge above and run all cells. The first cell installs the exact package versions
(NumPy 2.1.3, SciPy 1.16.3, autograd 1.9.1) and asks you to restart the runtime once; then run all cells
again. Result files are downloaded from this repository automatically.

**Locally:** clone the repository and run the notebook from its root (Python 3.13 recommended).

Two modes:

- **Check the analysis (minutes).** Run as is. Every stage loads its result file and recomputes all
  statistics, tables and figures from it. The last cell compares headline numbers with the paper.
- **Recompute everything (hours).** Set `RECOMPUTE = True` in the workspace cell and delete the result files
  (`runcache.json`, `stage{B,C,D}_cache.json`, `stageE/stageE_cache.json`). Every training run is recomputed
  from scratch. Approximate cost on 18 cores: Stage A ~1 h (single process), B 11 min, C 10 min, D 8 min,
  E 50 min; on a 2-core Colab runtime the parallel stages take about 9× longer.

## Reproducibility

- Training is deterministic given the seed and the package versions. Recomputing Stages A–D on a different
  machine reproduces **every run without LayerNorm bit for bit** (2,010 of 2,370 runs). The 360 runs of the
  LayerNorm arms differ in the 3rd–4th decimal, because LayerNorm's reductions round differently on
  different CPU/BLAS builds. No conclusion changes. Both sets are here: the original Colab files (root,
  checksums in `SHA256SUMS_colab_originals.txt`) and the local recomputation (`recomputed_local/`).
- Stage E is run from the registered code, `stageE_registered_code.zip`, whose SHA-256
  (`25ec7c72…3e52`) is part of the [OSF registration](https://osf.io/dgyuq). The notebook verifies it, and
  verifies that its own modules are byte-identical to the registered ones, before running Stage E.

## Files

| path | contents |
|---|---|
| `backbone_study.ipynb` | the notebook (all stages, figures, checks) |
| `runcache.json`, `stage{B,C,D}_cache.json` | Stage A–D results from the original Colab runs |
| `stageE/` | registered code, the 4,800-run Stage E results, the registered analysis output, the run log |
| `stageE_registered_code.zip` | the exact zip attached to the OSF registration |
| `registration/` | the registration text and the power analysis |
| `recomputed_local/` | Stage A–D results recomputed on a local machine (see Reproducibility) |
| `history/neural_selection_study_FINAL.ipynb` | the original notebook with its outputs, unchanged |
| `build_notebook.py` | rebuilds `backbone_study.ipynb` from the original notebook |

A note on history: an earlier version of this study described a locally hashed analysis plan as
"pre-registered". It had no external timestamp, so Stages A–D are treated as exploratory throughout; Stage E
is the only registered, confirmatory part.

## License

CC BY-NC 4.0 (see `LICENSE`), matching the OSF registration.

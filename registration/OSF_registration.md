# OSF Preregistration — Stage E of "Does the Backbone Choose the Equilibrium?"

Template: OSF Preregistration (https://osf.io/registries/osf/new → "OSF Preregistration").
Each heading below is one field of that form; paste the text under it. Attach the files listed in
"Attached files" and paste their SHA-256 digests where indicated.

---

## Study information

### Title
Confirmatory replication: does the policy architecture change the equilibrium selected by regularized
self-play? (Stage E)

### Authors
Luis Leal

### Description
An exploratory study (Stages A–D, 2,330 training runs; preprint/code: see "Other") compared policy
architectures in Regularized Nash Dynamics (R-NaD) with exact counterfactual values, on Kuhn poker and
three constructed variants whose Nash families are known in closed form. Two findings emerged from
comparisons that were designed after earlier data had been seen:

1. With parameter count matched to within 2% (attention d=11, 787 parameters vs MLP 770) and learning
   rate equalized (0.012), attention selects a lower equilibrium coordinate than the MLP: pooled over
   four games −0.00828, 95% CI [−0.01172, −0.00484] (Kuhn n=150/arm, other games n=40/arm).
2. On `kuhn_dupcall`, where the maximum-entropy split between two payoff-identical actions is exactly
   0.5, attention (d=8) converges to 0.4626 and 0.4486 on the two split coordinates (n=80), while the MLP
   reaches 0.5006 and 0.5000.

The analysis plan written during the exploratory study was not registered or timestamped, so none of
those results counts as confirmatory. Stage E tests both findings on new seeds with the same code and
hyperparameters, under the hypotheses, sample sizes and decision rules registered here.

### Hypotheses
- **H1 (primary).** The pooled matched contrast, mean(`attn_cap`) − mean(`mlp_lr012`) of the selected
  coordinate, over the four games {kuhn, kuhn_dupcall, kuhn_dupp1, kuhn_dupbet}, is negative.
  Directional prediction: negative; tested two-sided.
- **H2 (primary).** On `kuhn_dupcall`, the mean split of the `attn` arm is below 0.5 on both split
  coordinates (`split_Q`, `split_K`). Directional prediction: below 0.5; tested two-sided.

## Design plan

### Study type
Other: computational experiment (simulation study). No human or animal participants.

### Blinding
Not applicable. The analysis is fully scripted and is run once, unchanged, on the complete data.

### Study design
Between-arm comparison of neural policy architectures trained with the same solver. Each run trains one
policy network from one random seed and records its selected Nash coordinate, its exploitability and its
final policy. Arms (all at η=0.5, 1500 iterations, 3 inner steps, reference period 40, h=16):

| arm | architecture | learning rate | parameters (Kuhn) |
|---|---|---|---|
| `mlp_lr012` | MLP | 0.012 | 770 |
| `attn_cap` | single-head attention, d=11 | 0.012 | 787 |
| `mlp` | MLP | 0.015 | 770 |
| `attn` | single-head attention, d=8 | 0.012 | 490 |
| `attn_cap15` | single-head attention, d=11 | 0.015 | 787 |

Blocks (seed ranges are disjoint from every seed used before: 0–199 in Stages A–D, 900–911 pilot,
10000–10199 initialization control):

| block | games | arms | seeds | n per arm per game |
|---|---|---|---|---|
| H1 | kuhn, kuhn_dupcall, kuhn_dupp1, kuhn_dupbet | mlp_lr012, attn_cap | 50000–50399 | 400 |
| H2 | kuhn_dupcall | mlp, attn | 60000–60399 | 400 |
| S4 | kuhn | mlp, attn_cap15 | 70000–70399 | 400 |

Total runs: 4,800 (H1 3,200; H2 800; S4 800).

### Randomization
Seeds are consecutive integers in the registered ranges; each seed initializes the network
(`nbn.init_params(..., seed=s)`). Everything else is deterministic (exact game-tree values).

## Sampling plan

### Existing data
Registration prior to creation of data. No Stage-E run has been executed. The only executions of the
Stage-E code before registration are a two-seed smoke test on seeds 0 and 1, which are Stage-D seeds and
reproduce existing Stage-D records.

### Explanation of existing data
The exploratory Stages A–D exist and motivated these hypotheses; their standard deviations were used for
the power analysis. They are not part of the Stage-E analysis.

### Data collection procedures
`python stageE.py parallel <nproc>` runs every registered job; `python stageE.py merge` writes
`stageE_cache.json`. Runs are executed on a local Linux machine under Python 3.13, NumPy 2.1.3,
SciPy 1.16.3, autograd 1.9.1 (the versions used for Stages B–D).

### Sample size
H1: 400 seeds per arm per game (4 games × 2 arms). H2: 400 seeds per arm. S4: 400 seeds per arm.

### Sample size rationale
Computed by `power_stageE.py` from the Stage-C/D standard deviations (output attached).
- H1: smallest effect of interest |Δ| = 0.004 (about the instrument resolution 0.0047 and half the
  exploratory estimate, which is likely inflated because it is the result that prompted this test).
  At n=400, power = >0.999 at α=0.025 two-sided assuming independent games, 0.90 with
  between-game correlation 0.5.
- H2: smallest effect of interest 0.02 below 0.5 (about half the smaller exploratory deviation, 0.037).
  At n=400, power per coordinate = >0.999 (split_Q) and 0.976 (split_K) at α=0.0125.

### Stopping rule
All registered runs are executed; the analysis is run once, after all runs have finished. There is no
interim analysis. A run that crashes is re-run with the same seed (the computation is deterministic); a
run that completes but fails the convergence gate is excluded as specified below, not re-run.

## Variables

### Manipulated variables
Policy architecture (MLP vs attention), with parameter count and learning rate matched in H1.

### Measured variables
- `coord`: the selected coordinate read off the best-exploitability iterate (`game.selection_coord`):
  P(bet | Jack) on kuhn, kuhn_dupcall and kuhn_dupp1; total root bluff mass on kuhn_dupbet.
- `split_Q`, `split_K` (kuhn_dupcall): share of call mass on the first of two payoff-identical call
  actions, for the Queen and King information sets.
- `exploit`: exploitability (NashConv, exact best response) of the best iterate; `final_exploit`,
  `coord_final`: the same at the last iterate; `sigma`: the best-iterate policy.

### Indices
None beyond the above.

## Analysis plan

### Statistical models
Run as `python analyzeE.py stageE_cache.json` (attached; it is the analysis).
- **H1.** Per game: Welch t on `coord`, `attn_cap` minus `mlp_lr012`. Pooled: fixed-effect
  inverse-variance mean of the four per-game differences, z-test, two-sided. Equivalence: pooled TOST
  at ±0.010. Four-way verdict (DIFFERENT / EQUIVALENT / TRIVIALLY DIFFERENT / INCONCLUSIVE).
- **H2.** One-sample t of `attn` `split_Q` and `split_K` on kuhn_dupcall against 0.5; Holm over the two.

### Inference criteria
α = 0.025 for each of H1 and H2 (Bonferroni across the two primary hypotheses).
- H1 **SUPPORTED** if pooled p < 0.025 and the pooled estimate < 0. **CONTRADICTED** if p < 0.025 and
  the estimate > 0. Otherwise **NOT SUPPORTED**, labelled "equivalent within ±0.010" if the pooled TOST
  p < 0.025 and "inconclusive" otherwise.
- H2 **SUPPORTED** if both Holm-adjusted p < 0.025 and both means < 0.5; otherwise NOT SUPPORTED.

### Data exclusion
A run is excluded if its best-iterate exploitability is ≥ 0.02 or `numeric_ok` is false (the gate used
in Stages B–D). The number excluded per cell is reported. No other exclusions.

### Missing data
None expected (deterministic computation). Any registered run missing from the cache at analysis time is
reported as missing; the analysis is not run until all registered runs are present.

### Secondary analyses (declared; cannot overturn H1/H2)
- S1: Kuhn alone, matched contrast (H1 block). S2: H1 pooled with the final-iterate coordinate and a
  final-iterate gate. S3: Kuhn matched contrast with the least-squares projection metric. S4: Kuhn,
  `attn_cap15` − `mlp` (matched at learning rate 0.015). Holm across S1–S4 at α=0.05.
- S5: MLP split on kuhn_dupcall equivalent to 0.5 within ±0.02 (one-sample TOST).
- S6: trained minus untrained split on the same seeds (paired t), both arms, kuhn_dupcall.
- H1 pooled under assumed between-game correlation ρ ∈ {0, 0.3, 0.5, 0.7, 0.9} (descriptive).

### What the paper will report under each outcome
- H1 SUPPORTED: the architecture effect on selection is confirmed at the registered n; its size is
  reported with its interval and compared with the exploratory −0.00828.
- H1 NOT SUPPORTED (equivalent): the exploratory effect did not replicate, and the effect is bounded within
  ±0.010; the paper's headline becomes the bounded null plus the measurement audit.
- H1 NOT SUPPORTED (inconclusive): the effect is neither confirmed nor bounded; reported as such.
- H1 CONTRADICTED: reported as a sign reversal, added to the audit's list of fragile estimates.
- H2 SUPPORTED / NOT SUPPORTED: the split miscalibration is reported as confirmed / as an exploratory
  observation that did not replicate.
Every Stage-A–D result remains labelled exploratory in all cases.

### Exploratory analysis
Anything not listed above, if reported, is labelled exploratory.

## Other

### Attached files (SHA-256)
File `stageE_registered_code.zip`, SHA-256 `25ec7c72509fbbc4650033c6270370bd7c7c228ec5c31ea0ca490c6394e33e52`, containing:

```
bfc6a5ff7ceedfa6f271228db1ec6383fae957b06190f8fb4e5c75d834405511  analyzeE.py
de50850544893751eb594f4d92d9bcd667de273fdd68753208245917f8175a62  efg.py
8d8daebae500725a206f576f8094057ee7bae504e17ee7d76be089c59620ba97  games2.py
9b07660016c2391b3f3144e45867d379426eac043476718a8d76f52a1acf182c  nbn.py
5ef52b80e6d058211456f52ebca236f32c43da3efffd90838eb3adb7fd49709c  power_stageE.py
7f49bb7937904941ce2682a33f172b4c36782bfd7c311676469edf1c57b2dfd9  selstats.py
081e6687c64bf2cf4cf964e6b80dc96556e7aef0a9bd80545c54d3c08147e30f  stageE.py
90b067fc590f979c5c53f31674a7a53110fe2948668ca218a48762f4c08cfa24  stageE_config.json
85f99b3e31bf09ec892f18e901cce854bc7dbd7a73c165a231355ffca709370a  power_stageE_output.txt
ca60baacbda406212f6c4310afa0dd71529073a796f5bf29cdc2dd1151585b3d  README.txt
```

### Other
Exploratory code and data: the code and result caches of Stages B–E will be released with the preprint. Companion preprints: arXiv:2606.28308, arXiv:2607.17543.
Generative AI assistance: this registration and the Stage-E scripts were drafted with an LLM (Claude) and
reviewed by the author.

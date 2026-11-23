# Parity Verification for Data-Science/AI Ports: A Reproducible Methodology

> **Purpose.** This document records the method we used to verify that `pydaltoolbox`
> (a Python port) faithfully reproduces `daltoolbox` (the original R package), and
> generalizes it into a reusable protocol. It is written as the methodological backbone
> for a scientific article on *how to verify behavioral parity between two
> implementations of the same data-science/AI toolkit*. Case studies are drawn from the
> actual port; all are reproducible from `develop/parity/` in both repositories.

---

## 1. Problem statement

A **port** (or reimplementation, or refactor, or model re-platforming) claims to compute
"the same thing" as a reference implementation. In data science and machine learning this
claim is unusually hard to check, because:

1. **Stochasticity.** Train/test splits, random initializations, cross-validation folds,
   and stochastic optimizers make two *correct* runs disagree.
2. **Default divergence.** Two libraries wrapping "the same" algorithm often differ in
   defaults (pre-pruning, scaling, penalty strength, number of restarts) that silently
   change results.
3. **Backend numerics.** Different solvers (IRLS vs. L-BFGS), spline formulations, or
   floating-point paths produce different-but-valid outputs.
4. **Structural conventions.** Two libraries may define the *output set itself*
   differently (e.g., association rules with single- vs. multi-item consequents).

The goal of parity verification is to separate these four benign sources of disagreement
from the fifth, malignant one: **a defect in the port (or in the original) that makes the
two implementations compute genuinely different things.**

We define **behavioral parity** operationally: on *identical inputs*, the two
implementations produce outputs that are either (a) numerically equal up to a documented
tolerance, or (b) different only through an explicitly characterized, benign mechanism
(stochastic, default, numeric, or structural) that does not change the method's meaning.

---

## 2. The central hazard: confounding by input variance

The single most important methodological point: **you cannot compare two implementations
unless they see byte-for-byte the same data in the same order.**

Our first (naive) comparison ran each side with its own `set.seed`/`random_state` on its
own split. The R models appeared uniformly better — an artifact of *split variance*, not
of any real difference. Re-running on the same split dissolved the illusion entirely and
left exactly one true discrepancy (a decision-tree overfit).

**Protocol rule 1 — Freeze the input.** Generate the split (or the dataset, transaction
set, etc.) *once*, in one implementation, and **export it to a neutral interchange format
(CSV)**. Both sides read the same file. This removes RNG, ordering, and library-specific
sampling from the comparison. Everything downstream is then attributable to the
implementations themselves.

```
Python: split once -> write diabetes_train.csv / diabetes_test.csv
R:      read the same CSVs -> fit/evaluate on the identical rows
```

---

## 3. The comparison target: choose an output invariant

Different families expose different natural invariants. Pick the *coarsest output that
still reflects the method's decision*, so you are robust to irrelevant formatting while
sensitive to real behavior:

| Family                | Invariant compared                                   |
|-----------------------|------------------------------------------------------|
| Regression            | test R² on the shared split                           |
| Classification        | test accuracy on the shared split                     |
| Clustering            | external agreement with labels (entropy, purity)      |
| Feature selection     | the **set** of selected features                      |
| Pattern mining        | the itemset/rule set with support/confidence/lift     |
| Elbow/knee detectors  | the chosen index and value                            |

**Protocol rule 2 — Compare decisions, not representations.** Compare the selected feature
*set* (not column order), the partition *agreement* (not cluster id labels, which are
arbitrary), the rule *set* with its measures (not print order). This avoids thousands of
false positives from cosmetic differences (e.g., non-deterministic `frozenset` iteration
order produced spurious example-file diffs we had to discard).

---

## 4. A taxonomy of disagreement (and what to do about each)

When the invariants differ, classify the cause before "fixing" anything:

### 4.1 Exact parity
Outputs equal up to tolerance. Record it and lock it with a guard test (§6). Examples in
this project: regression R² across all learners; `corr`, `fss`, ReliefF, info-gain, and
Gaussian stepwise feature selection; apriori frequent itemsets and supports; PAM, c-means,
GMM, hierarchical clustering.

### 4.2 Benign — stochastic
Disagreement vanishes (or becomes small and mean-zero) when the randomness is controlled,
and neither side is "more correct." **Test for it by seeding or by repetition.** Example:
k-means with `nstart=1` landed on a poor local optimum in ~3/8 unseeded runs; with
multi-start it was stable. (Here the fix was still warranted — see §4.4 — because one
side's *default* made it fragile.)

### 4.3 Benign — numeric/solver, or penalty selection
Different but valid backends. **Confirm by finding a regime where they *do* agree.**
Examples:
- **Lasso feature selection** differs because the penalty `λ` is cross-validation-chosen
  (`cv.glmnet`, 10 random folds vs. `LassoCV`, 5 deterministic folds; different `λ`
  grids). Both selections are valid lasso solutions at slightly different penalties.
- **Binomial stepwise** on a quasi-separable dataset differs because the logistic
  log-likelihood → 0 under separation, making AIC ill-conditioned and sensitive to the
  optimizer. *We proved the stepwise logic itself is faithful* by running the **Gaussian**
  family on a non-separable dataset, where the two implementations select an identical
  feature set. This "find the agreeing regime" step is what distinguishes a benign numeric
  difference from a logic bug.

### 4.4 Benign-but-fixable — a weak default
The disagreement is benign in mechanism but one side's *default* produces materially worse
results. Faithfulness here means matching the *better* behavior, and the fix belongs on
the weaker side. Example: R `stats::kmeans` defaults to `nstart = 1`; we added
`nstart = 10` to match the robust multi-start the Python port already used.

### 4.5 Structural — differing output definitions
The two libraries define the output object differently. Characterize it precisely and
assert the shared sub-structure. Example: `mlxtend` emits association rules with
multi-item consequents (`{butter} => {bread, milk}`); R `arules::apriori` emits only
single-item consequents by default. **All single-consequent rules match exactly**
(support/confidence/lift); the extra rules are a documented backend property, not a defect.

### 4.6 Malignant — a real defect
A genuine behavioral bug in the port *or in the original*. Fix it on **both** sides
(asking the maintainer before editing the reference). Examples found this way:
- **Decision-tree overfit (port):** sklearn trees grow fully; R `tree::tree` pre-prunes
  via `mindev`. The port returned test R² ≈ 0 vs. R's 0.24. Fix: map `mindev` to
  `min_impurity_decrease = mindev · var(y)`.
- **GLM prediction inversion (original!):** R `cla_glm` modeled `P(Y = last factor level)`
  but assigned that probability to the *positive* class column unconditionally, inverting
  predictions whenever `positive` was not the last level (accuracy 0.04 instead of 0.96).
  Latent for years because the docstring example happened to put `positive` last. Fix:
  relevel the response. **Parity testing found a bug in the reference, not the port.**
- **DBSCAN `eps` estimation (port), two layered causes:** (i) an off-by-one — scikit-learn
  `NearestNeighbors` counts the query point itself, so `n_neighbors = k` yields the
  (k−1)-th true neighbor; R `dbscan::kNNdist(k)` excludes self. (ii) a deeper one — the
  elbow detector used a GCV smoothing spline whose wiggly second derivative put the "knee"
  at the tail (eps too large, purity 0.69). R used a *stiff* `smooth.spline(df = 2)`. We
  re-derived the detector via the Reinsch cubic-smoothing-spline with the penalty solved
  for effective `df = 2`, recovering the true knee (purity 0.92, matching R).

> **Ripple verification.** A fix to a shared primitive must be re-checked everywhere it is
> reused. The curvature detector also drives PCA component counts, pattern thresholds, and
> smoothing bin selection; after the rewrite we confirmed the smoothing bin count changed
> from 2 to 8 — **now identical to R** — i.e., the ripple was a fidelity *gain*, not a
> regression.

---

## 5. Bug localization: bisect the pipeline

When an invariant diverges, do **not** debug the whole pipeline. Peel it back to the
smallest mechanism:

1. **Confirm the gap** on the frozen input (rule out §2).
2. **Characterize** the gap numerically (e.g., `0.044 ≈ 1 − 0.956` → *inverted*, not
   *random*; a purity complement → a labeling flip; a value near the tail → wrong elbow).
3. **Find the agreeing regime** (another family, another dataset, another parameter) to
   isolate which component differs (§4.3).
4. **Instrument the single primitive** (print the intermediate: the estimated `eps`, the
   k-distance curve, the second-derivative curve, the chosen `λ`).
5. **Reproduce the reference's exact formula** and diff against it.

This is how the DBSCAN discrepancy was traced from "purity 0.69 vs 0.93" down to "the
curvature maximum is at index 145 vs 98" down to "GCV spline vs `smooth.spline(df=2)`."

---

## 6. Locking parity: guard tests as regression contracts

Every resolved comparison becomes a permanent test, so the parity cannot silently rot:

- **Exact invariants** are hard-coded as the contract (e.g., the R-verified selected
  feature sets, the itemset supports). These are safe when the underlying code is
  deterministic (native numpy, plain linear regression).
- **Benign-stochastic / numeric** cases get **loose** guards (thresholds, overlap, "not
  degenerate", "beats baseline") — tight enough to catch a collapse, loose enough to
  survive backend/version drift.
- **Structural** cases assert the shared sub-structure *and* the characterized difference
  (e.g., "all single-consequent rules match; extra rules all have |consequent| > 1").

These live in `tests/test_quality_*.py`. The parity *scripts* that produced them (one per
family, with their exported frozen inputs) live in `daltoolbox/develop/parity/` so any
reviewer can re-run the full comparison.

---

## 7. Threats to validity

- **Dataset difficulty.** Easy datasets (iris for classification) hide default/scaling
  bugs because every learner saturates. Prefer a dataset that *stresses* the mechanism:
  `breast_cancer` (heterogeneous feature scales) exposed scaling fidelity that iris masks.
- **Single-split luck.** A frozen split removes variance but may be unrepresentative;
  where cheap, repeat over several frozen splits.
- **Tolerance choice.** Too tight → flaky across versions; too loose → misses real
  regressions. Tie the tolerance to the mechanism (exact for deterministic code, loose for
  stochastic).
- **"Approximate" as a refuge.** Labeling a difference "benign" is a *claim that must be
  evidenced* (an agreeing regime, a seeded collapse, a formula). Do not use it to wave away
  an uninvestigated gap.
- **Reference is not ground truth.** The original can be wrong (we found an inversion bug
  in it). Parity means *agreement after both are correct*, not *the port matching the
  original's mistakes*.

---

## 8. The protocol, condensed

1. **Freeze** the input once; export to CSV; both sides read it.
2. **Choose** the coarsest decision-level invariant per family.
3. **Compare** decisions, not representations.
4. **Classify** each disagreement: exact / stochastic / numeric / weak-default /
   structural / defect.
5. **Evidence** every "benign" verdict (agreeing regime, seed, or formula).
6. **Localize** defects by bisecting to the smallest primitive and instrumenting it.
7. **Fix** defects on both sides (ask before editing the reference); **re-check the
   ripple** through every reuse of a shared primitive.
8. **Lock** the outcome with a guard test (exact or loose as warranted) and keep the
   runnable parity script alongside its frozen input.

---

## 9. Results of applying this protocol to `daltoolbox` → `pydaltoolbox`

| Family            | Verdict                                   | Defects found & fixed                                  |
|-------------------|-------------------------------------------|--------------------------------------------------------|
| Regression        | parity; 1 defect                          | tree overfit → `mindev` (port)                         |
| Classification    | parity; 1 defect                          | `cla_glm` prediction inversion (**original**)          |
| Clustering        | parity; 3 defects                         | kmeans `nstart` (original); dbscan eps off-by-one + curvature detector (port) |
| Feature selection | parity; no defects                        | lasso & binomial-stepwise benign (CV-λ / separation)   |
| Pattern mining    | parity; no defects                        | itemsets identical; rules differ only by multi-item consequents (structural) |

Five families; six genuine defects localized and fixed (two of them in the *reference*
implementation); the remainder of all disagreements explained as benign with evidence.
Each result is reproducible from the scripts and frozen inputs in
`daltoolbox/develop/parity/` and locked by `tests/test_quality_*.py`.

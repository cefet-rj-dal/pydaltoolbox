# pydaltoolbox

[![PyPI version](https://img.shields.io/pypi/v/pydaltoolbox.svg)](https://pypi.org/project/pydaltoolbox/)
[![Python versions](https://img.shields.io/pypi/pyversions/pydaltoolbox.svg)](https://pypi.org/project/pydaltoolbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python port of the R [DAL Toolbox](https://github.com/cefet-rj-dal/daltoolbox) —
*leveraging Experiment Lines for Data Analytics*.

pydaltoolbox organizes data analytics workflows (preprocessing, modeling,
evaluation, tuning, extensibility) around a consistent **Experiment Line** style
API. Its value is **not** reimplementing classic algorithms — it reuses
scikit-learn, numpy and pandas — but a **rigid, extensible interface** that wraps
them with a uniform `fit` / `action` / `evaluate` contract. Building a new method
means subclassing a base and implementing two or three methods; sampling,
evaluation and tuning come for free.

It is the foundation of the DAL Python family that also includes
[`pytspredit`](https://github.com/cefet-rj-dal/pytspredit) (time series prediction) and
[`pyharbinger`](https://github.com/cefet-rj-dal/pyharbinger) (time series event detection).

---

## Installation

Install the latest release from PyPI:

```bash
pip install pydaltoolbox
```

> **Beta:** the v1 foundation is released and new features are landing incrementally
> (one release at a time) on the way to the `1.0.0` launch. The badge above always shows
> the latest version. Feedback and issues are very welcome.

The development version is available on GitHub:

```bash
pip install git+https://github.com/cefet-rj-dal/pydaltoolbox.git
```

For a local, editable clone (for contributing or running the examples):

```bash
git clone https://github.com/cefet-rj-dal/pydaltoolbox.git
cd pydaltoolbox
pip install -e ".[dev]"
pytest
```

---

## First experiment

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)

split = dal.SampleRandom(seed=42).train_test(iris)
train, test = split["train"], split["test"]

model = dal.ClaDtree("Species", slevels, random_state=42).fit(train)

pred = model.predict(test.drop(columns="Species"))   # pass only the features
ev = model.evaluate(test["Species"], pred)
print(ev["metrics"])
```

---

## Documentation

- [Processed examples](examples/) — notebooks executed end to end, with real output
  embedded (the "processed documentation", analogous to the R package's `examples/`).
- [develop/planejamento.md](develop/planejamento.md) — the migration roadmap (phases, scope, status).
- [develop/deploy.md](develop/deploy.md) — step-by-step guide to publish the package to PyPI.

The example sources live under [notebooks/](notebooks/) (jupytext Markdown) and are
rendered into [examples/](examples/) by:

```bash
pip install -e ".[docs]"
python tools/build_examples.py
```

---

## Status

Beta — the v1 foundation is complete and Cycle 1 is extending coverage toward `1.0.0`
(see the [roadmap](develop/planejamento.md) for the live plan and version map). Implemented:

- **Core Experiment Line:** `DalBase`, `DalTransform`, `DalLearner`, `Predictor`,
  `set_params`, and the `adjust_*` helpers.
- **Sampling:** `SampleRandom`, `SampleStratified` (`train_test`, `k_fold`),
  `train_test_from_folds`.
- **Classification:** `Classification` base + `ClaDtree`, `ClaRpart`, `ClaKnn`,
  `ClaNb`, `ClaRf`, `ClaBagging`, `ClaBoosting`, `ClaSvm`, `ClaMlp`, `ClaGlm`,
  `ClaMultinom`, `ClaGlmnet`, `ClaMajority`.
- **Regression:** `Regression` base + `RegLm`, `RegDtree`, `RegKnn`, `RegRf`,
  `RegSvm`, `RegMlp`.
- **Clustering:** `Clusterer` base + metrics + `ClusterKmeans`, `ClusterDbscan`,
  `ClusterGmm`, `ClusterHclust`, `ClusterPam`, `ClusterCmeans` *(0.6.0)*.
- **Transformations:**
  - normalization: `MinMax`, `ZScore`
  - cleaning: `NaRemoval`, `ImputationSimple`, `OutliersBoxplot`, `OutliersGaussian`
  - predictive imputation: `ImputationPredictive`, `ImputationTree` *(0.3.0)*
  - encoding / projection: `CategMapping`, `DtPca`
  - feature selection: `FeatureSelectionCorr`, `Lasso`, `InfoGain`, `Fss`, `Stepwise`, `Relief`
  - feature engineering: `FeatureGeneration`, `Aggregation`, `HierarchyCut` *(0.4.0)*
  - smoothing: `SmoothingInter`, `SmoothingFreq`, `SmoothingQuantization`, `SmoothingCluster` *(0.2.0)*
  - balancing: `SampleBalance`, `BalSubsampling`, `BalOversampling`
  - sampling: `SampleSimple`, `SampleGroups` *(0.5.0)*
  - curvature helpers: `FitCurvatureMin`, `FitCurvatureMax`
- **Pattern mining:** `PatternMiner` base + `PatUtils` + `PatApriori` (frequent
  itemsets and association rules) *(0.9.0)*; rule filters `RuleFilter` / `RuleFilterNone` /
  `RuleFilterDara` + the `dara` divergent-association ranking *(0.10.0)*; `RuleFilterInterest`
  and support/confidence auto-estimation `SupportThreshold` / `ConfidenceThreshold`
  *(0.11.0)*. Needs the optional `[patterns]` extra (`pip install pydaltoolbox[patterns]`,
  brings `mlxtend`).
- **Tuning:** `DalTune` base + `ClaTune`, `RegTune`, `CluTune`.
- **Autoencoder bases:** `AutoencBaseE`, `AutoencBaseEd` (identity, for extension).
- **Reference datasets:** `datasets.iris()`, `datasets.diabetes()`, `datasets.boston()`.

Wrappers are thin adapters over scikit-learn / numpy / pandas, each reproducing the R
backend's default behavior (e.g. SVM standardizes inputs like `e1071`, trees pre-prune
like `tree`/`rpart`). See the [roadmap](develop/planejamento.md) for what is still coming —
pattern-mining rules filters and sequences, and the dependency-gated extras (XGBoost,
graph clustering, graphics).

---

## Related DAL projects

- [DAL Toolbox (R)](https://github.com/cefet-rj-dal/daltoolbox)
- [tspredit (R)](https://github.com/cefet-rj-dal/tspredit)
- [harbinger (R)](https://github.com/cefet-rj-dal/harbinger)
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

## License

MIT

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Feature Selection by Stepwise GLM (FeatureSelectionStepwise)

Stepwise selection fits a generalized linear model and adds and/or drops predictors one
at a time, each step taking the move that most lowers Akaike's Information Criterion
(`AIC = 2k - 2*logLik`). AIC rewards fit while penalizing model size. This mirrors the R
`feature_selection_stepwise` (`stats::step`): `direction` is `"forward"`, `"backward"`,
or `"both"`, and `family` is `"binomial"` (default, logistic — binary target) or
`"gaussian"` (linear — numeric target).

**Reference:** Akaike, H. (1974). *A New Look at the Statistical Model Identification.*
IEEE Transactions on Automatic Control.

## Binary target (binomial, default)

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris = iris.assign(is_versicolor=(iris["Species"] == "versicolor").astype(int))
iris = iris.drop(columns=["Species"])

fs = dal.FeatureSelectionStepwise("is_versicolor", direction="forward").fit(iris)
print("selected:", fs.selected)
fs.transform(iris).head()
```

## Numeric target (gaussian)

```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionStepwise("target", family="gaussian", direction="both").fit(diabetes)
print("selected:", fs.selected)
fs.transform(diabetes).head()
```

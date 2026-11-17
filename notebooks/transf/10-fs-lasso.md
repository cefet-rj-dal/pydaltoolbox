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

# Feature Selection by Lasso (FeatureSelectionLasso)

The lasso (L1-regularized regression) shrinks coefficients and drives some exactly to
zero, which doubles as embedded feature selection: the predictors kept are those with
non-zero coefficients at the cross-validated penalty. The target must be numeric.
Features are standardized first, matching glmnet's internal scaling.

**Reference:** Tibshirani, R. (1996). *Regression Shrinkage and Selection via the
Lasso.* Journal of the Royal Statistical Society, Series B.

```python
import pydaltoolbox as dal

diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionLasso("target").fit(diabetes)
fs.ranking
```

```python
print("selected:", fs.selected)
fs.transform(diabetes).head()
```

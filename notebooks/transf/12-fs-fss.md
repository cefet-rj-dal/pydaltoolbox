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

# Feature Selection by Forward Subset Search (FeatureSelectionFss)

Forward subset search starts from no predictors and adds, one at a time, the feature
that most improves the fit; it then keeps the subset size with the best adjusted
R-squared (which penalizes adding uninformative predictors). The target must be
numeric. This mirrors the R `feature_selection_fss` (`leaps::regsubsets`, forward).

**Reference:** Hocking, R. R. (1976). *The Analysis and Selection of Variables in
Linear Regression.* Biometrics.

```python
import pydaltoolbox as dal

diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionFss("target").fit(diabetes)
print("selected:", fs.selected)
```

```python
print("features before:", diabetes.shape[1] - 1, " after:", len(fs.selected))
fs.transform(diabetes).head()
```

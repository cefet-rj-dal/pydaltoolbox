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

# Feature Selection with Lasso (FeatureSelectionLasso)

## About the transformation
Lasso performs regression with an L1 penalty, shrinking some coefficients exactly to zero. This example keeps only the predictors with non-zero coefficients at the selected regularization level.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Lasso selection on the diabetes regression target
```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionLasso("target").fit(diabetes)
fs.transform(diabetes).head()
```

## References
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.*

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

# Feature Selection with Stepwise Search (FeatureSelectionStepwise)

## About the transformation
Stepwise search iteratively adds or removes predictors from a generalized linear model according to an information criterion. This example uses forward search for a regression target.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Forward stepwise selection on the diabetes regression target
```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionStepwise("target", family="gaussian", direction="forward").fit(diabetes)
fs.transform(diabetes).head()
```

## References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning.*

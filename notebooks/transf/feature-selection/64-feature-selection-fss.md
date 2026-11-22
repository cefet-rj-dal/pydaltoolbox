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

# Forward Stepwise Subset Search (FeatureSelectionFss)

## About the transformation
Forward stepwise subset search starts with no predictors and adds the feature that most improves the current regression model, keeping the subset with the best score.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Forward subset selection on the diabetes regression target
```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionFss("target").fit(diabetes)
fs.transform(diabetes).head()
```

## References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning.*

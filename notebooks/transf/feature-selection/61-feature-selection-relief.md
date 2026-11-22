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

# Feature Selection with Relief (FeatureSelectionRelief)

## About the transformation
Relief estimates feature relevance by comparing each observation with its nearest hit and nearest miss. Useful for ranking predictors in classification problems.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Rank features and keep the top 2
```python
iris = dal.datasets.iris()
fs = dal.FeatureSelectionRelief("Species", top=2, m=50, seed=1).fit(iris)
fs.transform(iris).head()
```

## References
- Kononenko, I. (1994). *Estimating attributes: Analysis and extensions of Relief.* ECML.

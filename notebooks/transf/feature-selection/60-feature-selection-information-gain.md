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

# Feature Selection with Information Gain (FeatureSelectionInfoGain)

## About the transformation
Information Gain measures the reduction in target entropy provided by each feature. This example ranks the predictors and keeps the most informative ones for a categorical target.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Rank features and keep the top 2
```python
iris = dal.datasets.iris()
fs = dal.FeatureSelectionInfoGain("Species", top=2).fit(iris)
fs.transform(iris).head()
```

## References
- Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning.* Morgan Kaufmann.

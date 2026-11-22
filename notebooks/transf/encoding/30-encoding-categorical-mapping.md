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

# Categorical Mapping — One-Hot (CategMapping)

## About the transformation
- `CategMapping`: converts a categorical column into binary (one-hot) variables, one output column per category.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Apply one-hot mapping to the `Species` column
```python
iris = dal.datasets.iris()
cm = dal.CategMapping("Species").fit(iris)
cm.transform(iris).head()
```

## References
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer. (1-of-K / one-hot encoding)

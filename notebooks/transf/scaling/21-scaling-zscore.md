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

# Z-Score Standardization (ZScore)

## About the transformation
- `ZScore`: standardizes numeric attributes to zero mean and unit variance. Can also rescale to a target mean (`nmean`) and standard deviation (`nsd`).
- For each numeric column: `(x - mean) / sd`, optionally `* nsd + nmean`.

## Didactic goal
Focus on what changes in the dataset after each operation. Useful to equalize scale across features and as input to scale-sensitive models.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Context and sample data
```python
iris = dal.datasets.iris()
iris.describe()
```

## Apply standard Z-Score (mean 0, sd 1) and inverse-transform
```python
norm = dal.ZScore().fit(iris)
ndata = norm.transform(iris)
ndata.describe()
```

## Inverse transform to recover the original scale
```python
norm.inverse_transform(ndata).describe()
```

## Standardization to a custom target mean and standard deviation
```python
norm = dal.ZScore(nmean=0.5, nsd=0.5 / 2.698).fit(iris)
ndata = norm.transform(iris)
ndata.describe()
```

## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Standardization)

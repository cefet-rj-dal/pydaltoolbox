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

# Outlier Removal — 3-sigma Rule (OutliersGaussian)

## About the transformation
- `OutliersGaussian`: flags as outliers values beyond mean +/- 3 standard deviations, assuming an approximately normal distribution.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Remove outliers using the 3-sigma rule and inspect
```python
iris = dal.datasets.iris().copy()
model = dal.OutliersGaussian(alpha=3.0).fit(iris)
clean = model.transform(iris)
print("rows before:", len(iris), " after:", len(clean))
clean.head()
```

## References
- Pukelsheim, F. (1994). *The Three Sigma Rule.* The American Statistician 48(2):88-91.

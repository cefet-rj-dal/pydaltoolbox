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

# Outlier Removal — Boxplot Rule (OutliersBoxplot)

## About the transformation
- `OutliersBoxplot`: identifies outliers by the boxplot rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR) and removes them from numeric attributes.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Remove outliers via the boxplot rule and inspect the result
```python
iris = dal.datasets.iris().copy()
model = dal.OutliersBoxplot(alpha=1.5).fit(iris)
clean = model.transform(iris)
print("rows before:", len(iris), " after:", len(clean))
clean.head()
```

## References
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.

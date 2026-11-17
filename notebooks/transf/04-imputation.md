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

# Simple Imputation (ImputationSimple)

Instead of dropping rows, imputation fills missing values with a simple statistic:
the median (default) or mean for numeric columns, and the mode for categorical ones.
The median is a robust default because it is less sensitive to outliers than the mean.
`fit` learns the per-column fill values; `transform` applies them.

**Reference:** Little, R. J. A., Rubin, D. B. (2019). *Statistical Analysis with
Missing Data.* Wiley.

```python
import numpy as np

import pydaltoolbox as dal

iris = dal.datasets.iris().copy()
iris["Species"] = iris["Species"].astype(object)
iris.loc[[1, 10, 25], "sepal_length"] = np.nan
iris.loc[[3, 15], "Species"] = np.nan
print("missing - sepal_length:", int(iris["sepal_length"].isna().sum()),
      "| Species:", int(iris["Species"].isna().sum()))
```

```python
imp = dal.ImputationSimple(method="median").fit(iris)
imp.values["sepal_length"], imp.values["Species"]
```

```python
out = imp.transform(iris)
print("missing after imputation:", int(out.isna().sum().sum()))
```

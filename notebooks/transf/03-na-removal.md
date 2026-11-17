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

# Missing-Value Removal (NaRemoval)

The simplest way to handle missing data is to drop any row that contains a missing
value. It is appropriate when missingness is rare and assumed to be random; otherwise
imputation is usually preferable because dropping rows discards information.
`NaRemoval` is a transform, so it slots into the same workflow as any other step.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Data Cleaning).

```python
import numpy as np

import pydaltoolbox as dal

iris = dal.datasets.iris().copy()
iris.loc[[1, 10, 25], "sepal_length"] = np.nan
print("rows with missing values:", int(iris["sepal_length"].isna().sum()))
```

```python
clean = dal.NaRemoval().transform(iris)
print("before:", len(iris), " after:", len(clean))
```

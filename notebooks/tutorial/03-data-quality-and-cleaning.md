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

# Tutorial 03 - Data Quality and Cleaning

Real projects rarely start on a clean table. Missing values, unusual observations, and
categorical attributes often need attention first. `pydaltoolbox` treats these steps as explicit
workflow operations.

## A small dataset with missing values
```python
import pandas as pd

import pydaltoolbox as dal

small = dal.datasets.iris().head(12).copy()
small.loc[[2, 7], "sepal_length"] = None
small.loc[4, "sepal_width"] = None
small
```

## Removing incomplete rows (a baseline cleaning operation)
```python
dal.NaRemoval().transform(small)
```

## Simple imputation (median for numeric, most frequent for categorical)
```python
imp = dal.ImputationSimple(method="median").fit(small)
imp.transform(small)
```

## A basic outlier strategy on numeric attributes
```python
iris = dal.datasets.iris()
out = dal.OutliersBoxplot().fit(iris)
clean = out.transform(iris)
print("rows before:", len(iris), " after:", len(clean))
clean.head()
```

## One-hot mapping of a categorical attribute
```python
cat_data = pd.DataFrame({"color": ["red", "blue", "green", "red"], "value": [10, 20, 15, 12]})
dal.CategMapping("color").fit(cat_data).transform(cat_data)
```

Many beginners jump straight to modeling; in practice, data-quality decisions shape the
experiment before the learner ever sees the data.

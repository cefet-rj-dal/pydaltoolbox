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

# Z-Score Normalization (ZScore)

Z-score normalization standardizes each numeric column to zero mean and unit
standard deviation using `(x - mean) / sd`, optionally rescaled to a target mean
(`nmean`) and sd (`nsd`). It is the standardization used, internally, by the SVM and
MLP wrappers; here it is available as an explicit pipeline step. Like all
transforms, it supports `fit`, `transform` and `inverse_transform`.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Standardization).

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris.head()
```

Fit learns the mean/sd of each numeric column; transform standardizes them.

```python
trans = dal.ZScore().fit(iris)
tiris = trans.transform(iris)
tiris.head()
```

Each numeric column now has (approximately) mean 0 and standard deviation 1.

```python
tiris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].describe().loc[["mean", "std"]]
```

`inverse_transform` recovers the original scale.

```python
trans.inverse_transform(tiris).head()
```

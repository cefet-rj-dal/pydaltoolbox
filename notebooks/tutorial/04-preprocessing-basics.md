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

# Tutorial 04 - Preprocessing Basics

After cleaning, make the attributes more suitable for modeling: scaling numeric variables and
reducing redundancy among predictors. Normalization changes the scale; correlation-based
selection simplifies the feature set.

## Original scale of the attributes
```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris.describe()
```

## Min-max normalization
```python
norm = dal.MinMax().fit(iris)
iris_norm = norm.transform(iris)
iris_norm.describe()
```

## Correlation-based feature selection
```python
fs = dal.FeatureSelectionCorr(cutoff=0.9).fit(iris_norm)
iris_fs = fs.transform(iris_norm)
print("before:", list(iris_norm.columns))
print("after: ", list(iris_fs.columns))
iris_fs.head()
```

Normalization prepares the scale; feature selection prepares the structure of the data.

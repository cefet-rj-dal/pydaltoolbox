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

# Categorical Mapping / One-Hot Encoding (CategMapping)

Many learners require numeric inputs, so categorical columns are commonly expanded
into one indicator (0/1) column per category -- one-hot encoding. `CategMapping`
replaces the target column with its indicator columns and keeps the remaining
columns, so it drops straight into a pipeline. `fit` records the categories, so the
same columns are produced for train and test even if a category is missing from one
of them.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Data Transformation).

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris.head()
```

```python
cm = dal.CategMapping("Species").fit(iris)
encoded = cm.transform(iris)
encoded.head()
```

The `Species` column is now three binary columns, while the numeric features are
preserved.

```python
encoded.tail()
```

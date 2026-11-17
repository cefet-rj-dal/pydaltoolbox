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

# Min-Max Normalization (MinMax)

Min-max normalization linearly rescales each numeric column to the `[0, 1]` range
using `(x - min) / (max - min)`. It is a common preprocessing step for
distance- and gradient-based learners (k-NN, SVM, neural networks), which are
sensitive to feature magnitude. As a `DalTransform`, it follows the full contract:
`fit` learns the per-column min/max, `transform` applies the scaling, and
`inverse_transform` recovers the original values.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Normalization).

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris.head()
```

Fit learns the min/max of each numeric column; transform rescales to `[0, 1]`.

```python
trans = dal.MinMax().fit(iris)
tiris = trans.transform(iris)
tiris.head()
```

`inverse_transform` reverses the scaling, recovering the original scale.

```python
back = trans.inverse_transform(tiris)
back.head()
```

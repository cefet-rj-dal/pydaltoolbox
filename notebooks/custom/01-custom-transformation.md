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

# Custom Transformation

The goal of this example is to show how to create a custom transformation that fits
naturally into the Experiment Line workflow. The integration contract is small and uniform:
define a constructor, store configuration, implement `fit`/`transform`, and keep the rest of
the analytical flow unchanged. Here the custom transformation integrates an external backend
(scikit-learn's power transform) under the DAL `DalTransform` contract.

## A custom transformation subclassing `DalTransform`
```python
import pandas as pd
from sklearn.preprocessing import PowerTransformer

import pydaltoolbox as dal


class YeoJohnson(dal.DalTransform):
    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, data, **kwargs):
        self.model = PowerTransformer(method="yeo-johnson").fit(data[self.columns])
        return self

    def transform(self, data, **kwargs):
        out = data.copy()
        out[self.columns] = self.model.transform(data[self.columns])
        return out
```

## Use it exactly like any built-in transformation
```python
iris = dal.datasets.iris()
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
tr = YeoJohnson(cols).fit(iris)
tr.transform(iris).describe()
```

## What to observe
- The data-mining method may live in an external backend, but the integration contract
  (`fit`/`transform`) stays small and predictable.
- Once the contract is implemented, the object drops into any DAL workflow unchanged.

## References
- Yeo, I.-K., and Johnson, R. A. (2000). *A new family of power transformations to improve
  normality or symmetry.* Biometrika.

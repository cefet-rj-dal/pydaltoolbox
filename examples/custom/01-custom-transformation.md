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




           sepal_length   sepal_width  petal_length   petal_width
    count  1.500000e+02  1.500000e+02  1.500000e+02  1.500000e+02
    mean   3.789561e-16  1.255292e-15 -2.368476e-16  9.473903e-17
    std    1.003350e+00  1.003350e+00  1.003350e+00  1.003350e+00
    min   -2.137770e+00 -2.759144e+00 -1.545592e+00 -1.476845e+00
    25%   -8.956896e-01 -5.614702e-01 -1.224374e+00 -1.189599e+00
    50%    2.642956e-02 -8.191725e-02  3.225908e-01  1.596788e-01
    75%    7.222371e-01  5.958605e-01  7.598052e-01  7.964903e-01
    max    2.176957e+00  2.743175e+00  1.828818e+00  1.658549e+00



## What to observe
- The data-mining method may live in an external backend, but the integration contract
  (`fit`/`transform`) stays small and predictable.
- Once the contract is implemented, the object drops into any DAL workflow unchanged.

## References
- Yeo, I.-K., and Johnson, R. A. (2000). *A new family of power transformations to improve
  normality or symmetry.* Biometrika.

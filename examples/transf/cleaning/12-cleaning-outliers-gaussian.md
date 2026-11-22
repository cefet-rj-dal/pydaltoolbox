# Outlier Removal — 3-sigma Rule (OutliersGaussian)

## About the transformation
- `OutliersGaussian`: flags as outliers values beyond mean +/- 3 standard deviations, assuming an approximately normal distribution.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Remove outliers using the 3-sigma rule and inspect


```python
iris = dal.datasets.iris().copy()
model = dal.OutliersGaussian(alpha=3.0).fit(iris)
clean = model.transform(iris)
print("rows before:", len(iris), " after:", len(clean))
clean.head()
```

    rows before: 150  after: 149
    




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



## References
- Pukelsheim, F. (1994). *The Three Sigma Rule.* The American Statistician 48(2):88-91.

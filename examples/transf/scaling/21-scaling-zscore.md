# Z-Score Standardization (ZScore)

## About the transformation
- `ZScore`: standardizes numeric attributes to zero mean and unit variance. Can also rescale to a target mean (`nmean`) and standard deviation (`nsd`).
- For each numeric column: `(x - mean) / sd`, optionally `* nsd + nmean`.

## Didactic goal
Focus on what changes in the dataset after each operation. Useful to equalize scale across features and as input to scale-sensitive models.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Context and sample data


```python
iris = dal.datasets.iris()
iris.describe()
```




           sepal_length  sepal_width  petal_length  petal_width
    count    150.000000   150.000000    150.000000   150.000000
    mean       5.843333     3.057333      3.758000     1.199333
    std        0.828066     0.435866      1.765298     0.762238
    min        4.300000     2.000000      1.000000     0.100000
    25%        5.100000     2.800000      1.600000     0.300000
    50%        5.800000     3.000000      4.350000     1.300000
    75%        6.400000     3.300000      5.100000     1.800000
    max        7.900000     4.400000      6.900000     2.500000



## Apply standard Z-Score (mean 0, sd 1) and inverse-transform


```python
norm = dal.ZScore().fit(iris)
ndata = norm.transform(iris)
ndata.describe()
```




           sepal_length   sepal_width  petal_length   petal_width
    count  1.500000e+02  1.500000e+02  1.500000e+02  1.500000e+02
    mean  -5.684342e-16 -7.815970e-16 -2.842171e-16 -3.789561e-16
    std    1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00
    min   -1.863780e+00 -2.425820e+00 -1.562342e+00 -1.442245e+00
    25%   -8.976739e-01 -5.903951e-01 -1.222456e+00 -1.179859e+00
    50%   -5.233076e-02 -1.315388e-01  3.353541e-01  1.320673e-01
    75%    6.722490e-01  5.567457e-01  7.602115e-01  7.880307e-01
    max    2.483699e+00  3.080455e+00  1.779869e+00  1.706379e+00



## Inverse transform to recover the original scale


```python
norm.inverse_transform(ndata).describe()
```




           sepal_length  sepal_width  petal_length  petal_width
    count    150.000000   150.000000    150.000000   150.000000
    mean       5.843333     3.057333      3.758000     1.199333
    std        0.828066     0.435866      1.765298     0.762238
    min        4.300000     2.000000      1.000000     0.100000
    25%        5.100000     2.800000      1.600000     0.300000
    50%        5.800000     3.000000      4.350000     1.300000
    75%        6.400000     3.300000      5.100000     1.800000
    max        7.900000     4.400000      6.900000     2.500000



## Standardization to a custom target mean and standard deviation


```python
norm = dal.ZScore(nmean=0.5, nsd=0.5 / 2.698).fit(iris)
ndata = norm.transform(iris)
ndata.describe()
```




           sepal_length  sepal_width  petal_length  petal_width
    count    150.000000   150.000000    150.000000   150.000000
    mean       0.500000     0.500000      0.500000     0.500000
    std        0.185322     0.185322      0.185322     0.185322
    min        0.154600     0.050441      0.210463     0.232720
    25%        0.333641     0.390587      0.273451     0.281346
    50%        0.490302     0.475623      0.562149     0.524475
    75%        0.624583     0.603177      0.640884     0.646040
    max        0.960285     1.070878      0.829850     0.816230



## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Standardization)

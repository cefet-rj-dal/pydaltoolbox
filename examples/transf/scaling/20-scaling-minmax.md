# Min-Max Normalization (MinMax)

## About the transformation
- `MinMax`: linearly rescales numeric attributes to [0, 1]. Useful for scale-sensitive algorithms and models that expect bounded inputs.
- For each numeric column: `(x - min) / (max - min)`; constant columns map to 0.

## Didactic goal
The value of the example is not only the call to `MinMax()`, but the before/after comparison. This is one of the most common steps before distance-based or scale-sensitive learners.

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



## Apply Min-Max and inspect the resulting scale


```python
norm = dal.MinMax().fit(iris)
ndata = norm.transform(iris)
ndata.describe()
```




           sepal_length  sepal_width  petal_length  petal_width
    count    150.000000   150.000000    150.000000   150.000000
    mean       0.428704     0.440556      0.467458     0.458056
    std        0.230018     0.181611      0.299203     0.317599
    min        0.000000     0.000000      0.000000     0.000000
    25%        0.222222     0.333333      0.101695     0.083333
    50%        0.416667     0.416667      0.567797     0.500000
    75%        0.583333     0.541667      0.694915     0.708333
    max        1.000000     1.000000      1.000000     1.000000



## Inverse transform (denormalize) to verify integrity


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



## What to observe
- Numeric attributes that originally had different ranges are now on a common scale.
- The categorical target remains untouched, which is usually what you want in supervised tasks.

## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Normalization)

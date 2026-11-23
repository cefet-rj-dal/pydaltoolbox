# Smoothing by Quantization (SmoothingQuantization)

## About the technique
- `SmoothingQuantization`: discretization/smoothing by one-dimensional k-means quantization, without using the class label.

## Didactic goal
Discretization turns a continuous variable into a small set of ordered bins; smoothing reduces noise by summarizing values into those bins. Choosing the number of bins is the key decision.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Apply the discretization and inspect the bins


```python
iris = dal.datasets.iris()
x = iris["sepal_length"]
sm = dal.SmoothingQuantization(n=2).fit(x)
bins = sm.transform(x)
pd.Series(bins).value_counts().sort_index()
```




    5.224096    83
    6.610448    67
    Name: count, dtype: int64



## Optimize the number of bins and refit


```python
tuned = dal.SmoothingQuantization(n=range(1, 21)).fit(iris["sepal_length"])
tuned.n
```




    8



## References
- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.*

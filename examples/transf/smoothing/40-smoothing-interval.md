# Smoothing by Interval (SmoothingInter)

## About the technique
- `SmoothingInter`: discretization/smoothing by regular (equal-width) intervals. Useful to summarize continuous variables into ranges.

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
sm = dal.SmoothingInter(n=2).fit(x)
bins = sm.transform(x)
pd.Series(bins).value_counts().sort_index()
```




    5.328421    95
    6.732727    55
    Name: count, dtype: int64



## Optimize the number of bins and refit


```python
tuned = dal.SmoothingInter(n=range(1, 21)).fit(iris["sepal_length"])
tuned.n
```




    8



## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Discretization)

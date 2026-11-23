# Smoothing by Clustering (SmoothingCluster)

## About the technique
- `SmoothingCluster`: discretization/smoothing by class-aware clustering, so the grouping is influenced by the target class.

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
sm = dal.SmoothingCluster("Species", n=3).fit(iris[["sepal_length", "Species"]])
bins = sm.transform(iris["sepal_length"])
pd.Series(bins).value_counts().sort_index()
```




    4.953846    52
    5.817021    47
    6.774510    51
    Name: count, dtype: int64



## Optimize the number of bins and refit


```python
tuned = dal.SmoothingCluster("Species", n=range(1, 9)).fit(iris[["sepal_length", "Species"]])
tuned.n
```




    7



## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Discretization)

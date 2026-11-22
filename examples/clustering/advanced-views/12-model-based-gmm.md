# Gaussian Mixture Model (ClusterGmm)

## About the method
- `ClusterGmm`: Gaussian mixture model clustering.

## Didactic goal
Keep the same clustering line of experiment and change only the clustering family to a probabilistic mixture model.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data; keep reference labels only for interpretation


```python
iris = dal.datasets.iris()
x = iris.drop(columns="Species")
ref = iris["Species"]
x.head()
```




       sepal_length  sepal_width  petal_length  petal_width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    2           4.7          3.2           1.3          0.2
    3           4.6          3.1           1.5          0.2
    4           5.0          3.6           1.4          0.2



## Model configuration


```python
model = dal.ClusterGmm(g=3, seed=42)
```

## Fit the model and obtain cluster labels


```python
model = model.fit(x)
clu = model.cluster(x)
pd.Series(clu).value_counts().sort_index()
```




    0    55
    1    50
    2    45
    Name: count, dtype: int64



## Evaluate the partition


```python
model.evaluate(clu, ref)["metrics"]
```




                    metric       value      goal      type
    0               loglik -180.196663  maximize  internal
    1              entropy    0.161149  minimize  external
    2               purity    0.966667  maximize  external
    3  adjusted_rand_index    0.903874  maximize  external



## References
- Fraley, C., and Raftery, A. E. (2002). *Model-Based Clustering, Discriminant Analysis, and Density Estimation.*

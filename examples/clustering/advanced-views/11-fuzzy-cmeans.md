# Fuzzy c-means (ClusterCmeans)

## About the method
- `ClusterCmeans`: fuzzy c-means clustering, which keeps membership degrees instead of only hard assignments.

## Didactic goal
Keep the same clustering line of experiment and change only the notion of assignment from hard clusters to soft memberships.

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
model = dal.ClusterCmeans(centers=3, m=2, seed=42)
```

## Fit the model and obtain cluster labels


```python
model = model.fit(x)
clu = model.cluster(x)
pd.Series(clu).value_counts().sort_index()
```




    0    50
    1    40
    2    60
    Name: count, dtype: int64



## Evaluate the partition


```python
model.evaluate(clu, ref)["metrics"]
```




                    metric      value      goal      type
    0           silhouette   0.549518  maximize  internal
    1          withinerror  60.505711  minimize  internal
    2              entropy   0.404097  minimize  external
    3               purity   0.893333  maximize  external
    4  adjusted_rand_index   0.729420  maximize  external



## Inspect the membership matrix attached to the result


```python
pd.DataFrame(model.membership_).head()
```




              0         1         2
    0  0.996624  0.001072  0.002304
    1  0.975853  0.007498  0.016650
    2  0.979826  0.006415  0.013760
    3  0.967427  0.010108  0.022465
    4  0.994470  0.001768  0.003762



## References
- Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms.*

# Fuzzy c-means Clustering (ClusterCmeans)

Fuzzy c-means is a *soft* clustering method: instead of assigning each point to a single
cluster, it gives every point a degree of membership in each of the `centers` clusters.
The fuzziness parameter `m` (m > 1) controls how soft the partition is. `cluster`
returns the hard assignment (the cluster of highest membership); the full membership
matrix is kept in the `membership_` attribute.

**Reference:** Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function
Algorithms.*


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterCmeans(centers=3, m=2, seed=42).fit(x)
labels = model.cluster(x)
```

Cross-tabulating the hard assignments against the true species:


```python
pd.crosstab(labels, iris["Species"])
```




    Species  setosa  versicolor  virginica
    row_0                                 
    0            50           0          0
    1             0           3         37
    2             0          47         13



The membership matrix holds, for each observation, how strongly it belongs to each
cluster (rows sum to 1):


```python
pd.DataFrame(model.membership_).head()
```




              0         1         2
    0  0.996624  0.001072  0.002304
    1  0.975853  0.007498  0.016650
    2  0.979826  0.006415  0.013760
    3  0.967427  0.010108  0.022465
    4  0.994470  0.001768  0.003762



`evaluate` reports internal quality (silhouette and the within-cluster fuzzy error) and,
when a reference label is given, external agreement (entropy, purity, adjusted Rand
index).


```python
model.evaluate(labels, iris["Species"])["metrics"]
```




                    metric      value      goal      type
    0           silhouette   0.549518  maximize  internal
    1          withinerror  60.505711  minimize  internal
    2              entropy   0.404097  minimize  external
    3               purity   0.893333  maximize  external
    4  adjusted_rand_index   0.729420  maximize  external



# PAM Clustering (ClusterPam)

PAM (Partitioning Around Medoids) groups the data around `k` representative data points
called *medoids*. Because a medoid is an actual observation rather than an average, PAM
is more robust to outliers than k-means. Clustering follows the same Experiment Line
contract: `fit`, then `cluster` to get labels, then `evaluate` to score the partition.

**Reference:** Kaufman, L. and Rousseeuw, P. J. (1990). *Finding Groups in Data: An
Introduction to Cluster Analysis.*


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterPam(k=3).fit(x)
labels = model.cluster(x)
```

Cross-tabulating the clusters against the true species shows how well the unsupervised
partition recovers the classes.


```python
pd.crosstab(labels, iris["Species"])
```




    Species  setosa  versicolor  virginica
    row_0                                 
    0            50           0          0
    1             0          48         14
    2             0           2         36



`evaluate` reports internal quality (silhouette, Davies-Bouldin, Calinski-Harabasz)
and, when a reference label is given, external agreement (entropy, purity, adjusted
Rand index).


```python
model.evaluate(labels, iris["Species"])["metrics"]
```




                    metric       value      goal      type
    0           silhouette    0.552819  maximize  internal
    1       davies_bouldin    0.661972  minimize  internal
    2    calinski_harabasz  561.627757  maximize  internal
    3              entropy    0.393886  minimize  external
    4               purity    0.893333  maximize  external
    5  adjusted_rand_index    0.730238  maximize  external



Unlike hierarchical or fuzzy clustering, a fitted PAM model can assign new, unseen
points -- each is sent to the nearest medoid.


```python
model.cluster(x.head(5))
```




    array([0, 0, 0, 0, 0])



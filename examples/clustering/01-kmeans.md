# k-means Clustering (ClusterKmeans)

k-means partitions the data into `k` groups by minimizing the within-cluster sum of
squares around `k` centroids. It is fast and a natural first choice for compact,
roughly spherical clusters. Clustering follows the same Experiment Line contract:
`fit`, then `cluster` to get labels, then `evaluate` to score the partition -- with
internal metrics (data only) and external metrics (against a known label).

**Reference:** MacQueen, J. (1967). *Some Methods for Classification and Analysis of
Multivariate Observations.* / Lloyd, S. (1982). *Least Squares Quantization in PCM.*


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterKmeans(k=3, seed=42).fit(x)
labels = model.cluster(x)
```

Cross-tabulating the clusters against the true species shows how well the unsupervised
partition recovers the classes.


```python
pd.crosstab(labels, iris["Species"])
```




    Species  setosa  versicolor  virginica
    row_0                                 
    0             0          48         14
    1            50           0          0
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



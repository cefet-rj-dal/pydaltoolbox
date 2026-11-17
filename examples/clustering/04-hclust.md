# Hierarchical Clustering (ClusterHclust)

Agglomerative hierarchical clustering builds a dendrogram by repeatedly merging the
closest groups; a flat partition is obtained by cutting the tree at `k` clusters (or at
a distance threshold `h`). The linkage (`method`) and distance (`dist`) control how
groups are compared, and features are standardized first by default.

**Reference:** Johnson, S. C. (1967). *Hierarchical Clustering Schemes.* Psychometrika.


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterHclust(k=3, method="ward").fit(x)
labels = model.cluster(x)
pd.crosstab(labels, iris["Species"])
```




    Species  setosa  versicolor  virginica
    row_0                                 
    0             0          23         48
    1            49           0          0
    2             1          27          2




```python
model.evaluate(labels, iris["Species"])["metrics"]
```




                    metric       value      goal      type
    0           silhouette    0.503050  maximize  internal
    1       davies_bouldin    0.682640  minimize  internal
    2    calinski_harabasz  430.902102  maximize  internal
    3              entropy    0.542245  minimize  external
    4               purity    0.826667  maximize  external
    5  adjusted_rand_index    0.615323  maximize  external



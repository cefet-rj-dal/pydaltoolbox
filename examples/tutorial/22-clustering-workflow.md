# Tutorial 22 - Clustering Workflow

Clustering has no explicit target during training, but the workflow is still structured: choose
a method, fit it, obtain assignments, and inspect the result. Preprocessing can change
unsupervised results substantially.

## K-means on the original scale


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterKmeans(k=3, seed=1).fit(x)
clu = model.cluster(x)
print(pd.Series(clu).value_counts().sort_index().to_dict())
model.evaluate(clu, iris["Species"])["metrics"]
```

    {0: 62, 1: 50, 2: 38}
    




                    metric       value      goal      type
    0           silhouette    0.552819  maximize  internal
    1       davies_bouldin    0.661972  minimize  internal
    2    calinski_harabasz  561.627757  maximize  internal
    3              entropy    0.393886  minimize  external
    4               purity    0.893333  maximize  external
    5  adjusted_rand_index    0.730238  maximize  external



## Normalize and repeat the same clustering


```python
norm = dal.MinMax().fit(iris)
x_norm = norm.transform(iris).drop(columns="Species")

model_norm = dal.ClusterKmeans(k=3, seed=1).fit(x_norm)
clu_norm = model_norm.cluster(x_norm)
print(pd.Series(clu_norm).value_counts().sort_index().to_dict())
model_norm.evaluate(clu_norm, iris["Species"])["metrics"]
```

    {0: 61, 1: 50, 2: 39}
    




                    metric       value      goal      type
    0           silhouette    0.504769  maximize  internal
    1       davies_bouldin    0.760277  minimize  internal
    2    calinski_harabasz  359.845074  maximize  internal
    3              entropy    0.417766  minimize  external
    4               purity    0.886667  maximize  external
    5  adjusted_rand_index    0.716342  maximize  external



In unsupervised learning, the data representation can matter as much as the algorithm.

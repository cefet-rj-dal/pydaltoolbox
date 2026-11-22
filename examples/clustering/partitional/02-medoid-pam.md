# PAM — Partitioning Around Medoids (ClusterPam)

## About the method
- `ClusterPam`: Partitioning Around Medoids. Similar to k-means, but uses medoids (actual observations) instead of centroids.

## Didactic goal
Keep the same clustering line of experiment and change only the partitioning strategy.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal
from pydaltoolbox.clustering.metrics import (
    metric_davies_bouldin,
    metric_entropy,
    metric_purity,
    metric_silhouette,
)

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
model = dal.ClusterPam(k=3)
model.eval_internal = [metric_silhouette, metric_davies_bouldin]
model.eval_external = [metric_entropy, metric_purity]
```

## Fit the model and obtain cluster labels


```python
model = model.fit(x)
clu = model.cluster(x)
pd.Series(clu).value_counts().sort_index()
```




    0    50
    1    62
    2    38
    Name: count, dtype: int64



## Evaluate the partition


```python
model.evaluate(clu, ref)["metrics"]
```




               metric     value      goal      type
    0      silhouette  0.552819  maximize  internal
    1  davies_bouldin  0.661972  minimize  internal
    2         entropy  0.393886  minimize  external
    3          purity  0.893333  maximize  external



## Assign new, unseen points to the nearest medoid


```python
model.cluster(x.head(5))
```




    array([0, 0, 0, 0, 0])



## What to observe
- The workflow is unchanged from k-means.
- The method-specific difference is that medoids are actual observations, which can make the partition more robust to outliers.

## References
- Kaufman, L., and Rousseeuw, P. J. (1990). *Finding Groups in Data.*
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*
- Davies, D. L., and Bouldin, D. W. (1979). *A cluster separation measure.*

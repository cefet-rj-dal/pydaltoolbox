# DBSCAN (ClusterDbscan)

## About the method
- `ClusterDbscan`: density-based clustering that identifies dense regions and can label sparse points as noise (`-1`).

## Didactic goal
Keep the same clustering line of experiment and change only the grouping principle, from partitioning all cases to detecting dense neighborhoods.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal
from pydaltoolbox.clustering.metrics import metric_entropy, metric_purity

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
model = dal.ClusterDbscan(min_pts=3)
model.eval_external = [metric_entropy, metric_purity]
```

## Fit the model and obtain cluster labels


```python
model = model.fit(x)
clu = model.cluster(x)
pd.Series(clu).value_counts().sort_index()
```




    -1     5
     0    49
     1    92
     2     4
    Name: count, dtype: int64



## Evaluate the partition


```python
model.evaluate(clu, ref)["metrics"]
```




             metric     value      goal      type
    0  noise_points  5.000000  minimize  internal
    1       entropy  0.637398  minimize  external
    2        purity  0.686667  maximize  external



## What to observe
- The workflow is the same as in the partition-based methods.
- The method-specific difference is that DBSCAN may leave some cases as noise instead of forcing all observations into clusters.

## References
- Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.*
- Zhao, Y., and Karypis, G. (2001). *Criterion functions for document clustering.*

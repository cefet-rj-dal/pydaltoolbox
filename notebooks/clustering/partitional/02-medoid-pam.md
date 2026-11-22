---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---



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

## Evaluate the partition
```python
model.evaluate(clu, ref)["metrics"]
```

## Assign new, unseen points to the nearest medoid
```python
model.cluster(x.head(5))
```

## What to observe
- The workflow is unchanged from k-means.
- The method-specific difference is that medoids are actual observations, which can make the partition more robust to outliers.

## References
- Kaufman, L., and Rousseeuw, P. J. (1990). *Finding Groups in Data.*
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*
- Davies, D. L., and Bouldin, D. W. (1979). *A cluster separation measure.*

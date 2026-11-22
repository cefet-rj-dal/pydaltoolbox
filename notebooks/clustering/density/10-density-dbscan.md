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

## Evaluate the partition
```python
model.evaluate(clu, ref)["metrics"]
```

## What to observe
- The workflow is the same as in the partition-based methods.
- The method-specific difference is that DBSCAN may leave some cases as noise instead of forcing all observations into clusters.

## References
- Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.*
- Zhao, Y., and Karypis, G. (2001). *Criterion functions for document clustering.*

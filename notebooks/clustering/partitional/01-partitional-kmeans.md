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



# k-means Clustering (ClusterKmeans)


## About the method
- `ClusterKmeans`: partitions data into `k` groups by minimizing within-cluster variance.

## Didactic goal
Establish the standard clustering line of experiment used throughout this family. Later examples should be read as changes in the clustering method, not as changes in the workflow.

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
model = dal.ClusterKmeans(k=3, seed=42)
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

## What to observe
- The labels in `ref` are not used during fitting; they only help interpret the partition afterward.
- This is the reference clustering workflow used by the other examples in this family.

## References
- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.*
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*
- Davies, D. L., and Bouldin, D. W. (1979). *A cluster separation measure.*
- Zhao, Y., and Karypis, G. (2001). *Criterion functions for document clustering.*

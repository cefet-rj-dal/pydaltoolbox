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

`evaluate` reports internal quality (silhouette, Davies-Bouldin, Calinski-Harabasz)
and, when a reference label is given, external agreement (entropy, purity, adjusted
Rand index).

```python
model.evaluate(labels, iris["Species"])["metrics"]
```

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

`evaluate` reports internal quality (silhouette, Davies-Bouldin, Calinski-Harabasz)
and, when a reference label is given, external agreement (entropy, purity, adjusted
Rand index).

```python
model.evaluate(labels, iris["Species"])["metrics"]
```

Unlike hierarchical or fuzzy clustering, a fitted PAM model can assign new, unseen
points -- each is sent to the nearest medoid.

```python
model.cluster(x.head(5))
```

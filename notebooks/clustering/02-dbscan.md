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

DBSCAN finds clusters as dense regions separated by sparse areas, and labels isolated
points as noise (`-1`). It does not need the number of clusters in advance, but it is
sensitive to `eps` (neighborhood radius) and `min_pts`. When `eps` is not given it is
estimated from the elbow of the sorted k-nearest-neighbor distance curve, as in the R
package. Its main quality metric is the number of noise points.

**Reference:** Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). *A Density-Based
Algorithm for Discovering Clusters in Large Spatial Databases with Noise.* KDD.

```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterDbscan(min_pts=5).fit(x)
labels = model.cluster(x)
print("estimated eps:", round(model.eps, 4))
pd.Series(labels).value_counts()
```

```python
model.evaluate(labels, iris["Species"])["metrics"]
```

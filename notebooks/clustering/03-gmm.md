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

# Gaussian Mixture Model (ClusterGmm)

A Gaussian mixture models the data as a weighted sum of Gaussian components and assigns
each point to its most likely component -- a soft, probabilistic alternative to k-means
that can capture elliptical clusters. When the number of components `g` is not given, it
is chosen by minimizing the BIC (the mclust-style model selection). Its main quality
metric is the log-likelihood.

**Reference:** Fraley, C., Raftery, A. E. (2002). *Model-Based Clustering,
Discriminant Analysis, and Density Estimation.* JASA.

```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterGmm(seed=42).fit(x)
labels = model.cluster(x)
print("components chosen (BIC):", model.g)
pd.crosstab(labels, iris["Species"])
```

```python
model.evaluate(labels, iris["Species"])["metrics"]
```

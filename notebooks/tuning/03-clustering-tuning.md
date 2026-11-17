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

# Clustering Tuning (CluTune)

Clustering is unsupervised, so `CluTune` does not use cross-validation. Instead it fits
the clusterer for each configuration and scores it with the model's intrinsic metric
(for k-means, the silhouette), then the model's selector picks the best configuration.
`fit` returns the tuned, fitted clusterer.

**Reference:** Satopaa, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). *Finding a
"Kneedle" in a Haystack: Detecting Knee Points in System Behavior.*

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

tune = dal.CluTune(dal.ClusterKmeans(seed=42), ranges={"k": range(2, 8)})
model = tune.fit(x)
print("chosen k:", model.k)
```

```python
labels = model.cluster(x)
model.evaluate(labels, iris["Species"])["metrics"]
```

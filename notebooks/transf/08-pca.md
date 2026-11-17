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

# Principal Component Analysis (DtPca)

PCA is an unsupervised dimensionality-reduction technique: it rotates correlated
features into orthogonal components ordered by explained variance, so the first few
often capture most of the information. `DtPca` standardizes the numeric predictors
first (center + scale), excludes the target `attribute` from the projection and
reattaches it afterwards. If `components` is not given, it is chosen automatically by
a minimum-curvature elbow on the cumulative explained variance.

**Reference:** Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of
Points in Space.* / Hotelling, H. (1933). *Analysis of a Complex of Statistical
Variables into Principal Components.*

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

Automatic component selection (the elbow rule picks the number of components):

```python
pca = dal.DtPca("Species").fit(iris)
print("components chosen:", pca.components)
pca.transform(iris).head()
```

You can also set the number of components explicitly:

```python
dal.DtPca("Species", components=2).fit(iris).transform(iris).head()
```

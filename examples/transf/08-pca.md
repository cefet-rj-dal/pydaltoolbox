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

    components chosen: 2
    




            PC1       PC2 Species
    0 -2.264703  0.480027  setosa
    1 -2.080961 -0.674134  setosa
    2 -2.364229 -0.341908  setosa
    3 -2.299384 -0.597395  setosa
    4 -2.389842  0.646835  setosa



You can also set the number of components explicitly:


```python
dal.DtPca("Species", components=2).fit(iris).transform(iris).head()
```




            PC1       PC2 Species
    0 -2.264703  0.480027  setosa
    1 -2.080961 -0.674134  setosa
    2 -2.364229 -0.341908  setosa
    3 -2.299384 -0.597395  setosa
    4 -2.389842  0.646835  setosa



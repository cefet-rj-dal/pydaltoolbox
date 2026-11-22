# Principal Component Analysis (DtPca)

## About the transformation
- `DtPca`: Principal Component Analysis projects correlated variables onto orthogonal components ordered by explained variance. You can let the tool pick the number of components via an elbow heuristic or set it explicitly.

## Didactic goal
Focus on what changes in the dataset after each operation.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Fit PCA using the target column as reference and transform


```python
iris = dal.datasets.iris()
mypca = dal.DtPca("Species").fit(iris)
mypca.transform(iris).head()
```




            PC1       PC2 Species
    0 -2.264703  0.480027  setosa
    1 -2.080961 -0.674134  setosa
    2 -2.364229 -0.341908  setosa
    3 -2.299384 -0.597395  setosa
    4 -2.389842  0.646835  setosa



## Manually set the number of components and repeat


```python
dal.DtPca("Species", components=3).fit(iris).transform(iris).head()
```




            PC1       PC2       PC3 Species
    0 -2.264703  0.480027  0.127706  setosa
    1 -2.080961 -0.674134  0.234609  setosa
    2 -2.364229 -0.341908 -0.044201  setosa
    3 -2.299384 -0.597395 -0.091290  setosa
    4 -2.389842  0.646835 -0.015738  setosa



## References
- Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.*
- Hotelling, H. (1933). *Analysis of a complex of statistical variables into principal components.*

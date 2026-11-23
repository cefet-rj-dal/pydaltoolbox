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
    0 -2.257141  0.478424  setosa
    1 -2.074013 -0.671883  setosa
    2 -2.356335 -0.340766  setosa
    3 -2.291707 -0.595400  setosa
    4 -2.381863  0.644676  setosa



## Manually set the number of components and repeat


```python
dal.DtPca("Species", components=3).fit(iris).transform(iris).head()
```




            PC1       PC2       PC3 Species
    0 -2.257141  0.478424  0.127280  setosa
    1 -2.074013 -0.671883  0.233826  setosa
    2 -2.356335 -0.340766 -0.044054  setosa
    3 -2.291707 -0.595400 -0.090985  setosa
    4 -2.381863  0.644676 -0.015686  setosa



## References
- Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.*
- Hotelling, H. (1933). *Analysis of a complex of statistical variables into principal components.*

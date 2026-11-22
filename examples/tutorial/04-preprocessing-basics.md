# Tutorial 04 - Preprocessing Basics

After cleaning, make the attributes more suitable for modeling: scaling numeric variables and
reducing redundancy among predictors. Normalization changes the scale; correlation-based
selection simplifies the feature set.

## Original scale of the attributes


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris.describe()
```




           sepal_length  sepal_width  petal_length  petal_width
    count    150.000000   150.000000    150.000000   150.000000
    mean       5.843333     3.057333      3.758000     1.199333
    std        0.828066     0.435866      1.765298     0.762238
    min        4.300000     2.000000      1.000000     0.100000
    25%        5.100000     2.800000      1.600000     0.300000
    50%        5.800000     3.000000      4.350000     1.300000
    75%        6.400000     3.300000      5.100000     1.800000
    max        7.900000     4.400000      6.900000     2.500000



## Min-max normalization


```python
norm = dal.MinMax().fit(iris)
iris_norm = norm.transform(iris)
iris_norm.describe()
```




           sepal_length  sepal_width  petal_length  petal_width
    count    150.000000   150.000000    150.000000   150.000000
    mean       0.428704     0.440556      0.467458     0.458056
    std        0.230018     0.181611      0.299203     0.317599
    min        0.000000     0.000000      0.000000     0.000000
    25%        0.222222     0.333333      0.101695     0.083333
    50%        0.416667     0.416667      0.567797     0.500000
    75%        0.583333     0.541667      0.694915     0.708333
    max        1.000000     1.000000      1.000000     1.000000



## Correlation-based feature selection


```python
fs = dal.FeatureSelectionCorr(cutoff=0.9).fit(iris_norm)
iris_fs = fs.transform(iris_norm)
print("before:", list(iris_norm.columns))
print("after: ", list(iris_fs.columns))
iris_fs.head()
```

    before: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Species']
    after:  ['sepal_length', 'sepal_width', 'petal_width', 'Species']
    




       sepal_length  sepal_width  petal_width Species
    0      0.222222     0.625000     0.041667  setosa
    1      0.166667     0.416667     0.041667  setosa
    2      0.111111     0.500000     0.041667  setosa
    3      0.083333     0.458333     0.041667  setosa
    4      0.194444     0.666667     0.041667  setosa



Normalization prepares the scale; feature selection prepares the structure of the data.

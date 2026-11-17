# Z-Score Normalization (ZScore)

Z-score normalization standardizes each numeric column to zero mean and unit
standard deviation using `(x - mean) / sd`, optionally rescaled to a target mean
(`nmean`) and sd (`nsd`). It is the standardization used, internally, by the SVM and
MLP wrappers; here it is available as an explicit pipeline step. Like all
transforms, it supports `fit`, `transform` and `inverse_transform`.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Standardization).


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris.head()
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



Fit learns the mean/sd of each numeric column; transform standardizes them.


```python
trans = dal.ZScore().fit(iris)
tiris = trans.transform(iris)
tiris.head()
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0     -0.897674     1.015602     -1.335752    -1.311052  setosa
    1     -1.139200    -0.131539     -1.335752    -1.311052  setosa
    2     -1.380727     0.327318     -1.392399    -1.311052  setosa
    3     -1.501490     0.097889     -1.279104    -1.311052  setosa
    4     -1.018437     1.245030     -1.335752    -1.311052  setosa



Each numeric column now has (approximately) mean 0 and standard deviation 1.


```python
tiris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].describe().loc[["mean", "std"]]
```




          sepal_length   sepal_width  petal_length   petal_width
    mean -5.684342e-16 -7.815970e-16 -2.842171e-16 -3.789561e-16
    std   1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00



`inverse_transform` recovers the original scale.


```python
trans.inverse_transform(tiris).head()
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



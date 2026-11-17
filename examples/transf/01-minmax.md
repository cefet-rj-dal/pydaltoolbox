# Min-Max Normalization (MinMax)

Min-max normalization linearly rescales each numeric column to the `[0, 1]` range
using `(x - min) / (max - min)`. It is a common preprocessing step for
distance- and gradient-based learners (k-NN, SVM, neural networks), which are
sensitive to feature magnitude. As a `DalTransform`, it follows the full contract:
`fit` learns the per-column min/max, `transform` applies the scaling, and
`inverse_transform` recovers the original values.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Normalization).


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



Fit learns the min/max of each numeric column; transform rescales to `[0, 1]`.


```python
trans = dal.MinMax().fit(iris)
tiris = trans.transform(iris)
tiris.head()
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0      0.222222     0.625000      0.067797     0.041667  setosa
    1      0.166667     0.416667      0.067797     0.041667  setosa
    2      0.111111     0.500000      0.050847     0.041667  setosa
    3      0.083333     0.458333      0.084746     0.041667  setosa
    4      0.194444     0.666667      0.067797     0.041667  setosa



`inverse_transform` reverses the scaling, recovering the original scale.


```python
back = trans.inverse_transform(tiris)
back.head()
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



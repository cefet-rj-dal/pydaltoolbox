# Categorical Mapping / One-Hot Encoding (CategMapping)

Many learners require numeric inputs, so categorical columns are commonly expanded
into one indicator (0/1) column per category -- one-hot encoding. `CategMapping`
replaces the target column with its indicator columns and keeps the remaining
columns, so it drops straight into a pipeline. `fit` records the categories, so the
same columns are produced for train and test even if a category is missing from one
of them.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Data Transformation).


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




```python
cm = dal.CategMapping("Species").fit(iris)
encoded = cm.transform(iris)
encoded.head()
```




       sepal_length  sepal_width  petal_length  petal_width  Species_setosa  Species_versicolor  Species_virginica
    0           5.1          3.5           1.4          0.2               1                   0                  0
    1           4.9          3.0           1.4          0.2               1                   0                  0
    2           4.7          3.2           1.3          0.2               1                   0                  0
    3           4.6          3.1           1.5          0.2               1                   0                  0
    4           5.0          3.6           1.4          0.2               1                   0                  0



The `Species` column is now three binary columns, while the numeric features are
preserved.


```python
encoded.tail()
```




         sepal_length  sepal_width  petal_length  petal_width  Species_setosa  Species_versicolor  Species_virginica
    145           6.7          3.0           5.2          2.3               0                   0                  1
    146           6.3          2.5           5.0          1.9               0                   0                  1
    147           6.5          3.0           5.2          2.0               0                   0                  1
    148           6.2          3.4           5.4          2.3               0                   0                  1
    149           5.9          3.0           5.1          1.8               0                   0                  1



# Tutorial 03 - Data Quality and Cleaning

Real projects rarely start on a clean table. Missing values, unusual observations, and
categorical attributes often need attention first. `pydaltoolbox` treats these steps as explicit
workflow operations.

## A small dataset with missing values


```python
import pandas as pd

import pydaltoolbox as dal

small = dal.datasets.iris().head(12).copy()
small.loc[[2, 7], "sepal_length"] = None
small.loc[4, "sepal_width"] = None
small
```




        sepal_length  sepal_width  petal_length  petal_width Species
    0            5.1          3.5           1.4          0.2  setosa
    1            4.9          3.0           1.4          0.2  setosa
    2            NaN          3.2           1.3          0.2  setosa
    3            4.6          3.1           1.5          0.2  setosa
    4            5.0          NaN           1.4          0.2  setosa
    5            5.4          3.9           1.7          0.4  setosa
    6            4.6          3.4           1.4          0.3  setosa
    7            NaN          3.4           1.5          0.2  setosa
    8            4.4          2.9           1.4          0.2  setosa
    9            4.9          3.1           1.5          0.1  setosa
    10           5.4          3.7           1.5          0.2  setosa
    11           4.8          3.4           1.6          0.2  setosa



## Removing incomplete rows (a baseline cleaning operation)


```python
dal.NaRemoval().transform(small)
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.6          3.1           1.5          0.2  setosa
    3           5.4          3.9           1.7          0.4  setosa
    4           4.6          3.4           1.4          0.3  setosa
    5           4.4          2.9           1.4          0.2  setosa
    6           4.9          3.1           1.5          0.1  setosa
    7           5.4          3.7           1.5          0.2  setosa
    8           4.8          3.4           1.6          0.2  setosa



## Simple imputation (median for numeric, most frequent for categorical)


```python
imp = dal.ImputationSimple(method="median").fit(small)
imp.transform(small)
```




        sepal_length  sepal_width  petal_length  petal_width Species
    0            5.1          3.5           1.4          0.2  setosa
    1            4.9          3.0           1.4          0.2  setosa
    2            4.9          3.2           1.3          0.2  setosa
    3            4.6          3.1           1.5          0.2  setosa
    4            5.0          3.4           1.4          0.2  setosa
    5            5.4          3.9           1.7          0.4  setosa
    6            4.6          3.4           1.4          0.3  setosa
    7            4.9          3.4           1.5          0.2  setosa
    8            4.4          2.9           1.4          0.2  setosa
    9            4.9          3.1           1.5          0.1  setosa
    10           5.4          3.7           1.5          0.2  setosa
    11           4.8          3.4           1.6          0.2  setosa



## A basic outlier strategy on numeric attributes


```python
iris = dal.datasets.iris()
out = dal.OutliersBoxplot().fit(iris)
clean = out.transform(iris)
print("rows before:", len(iris), " after:", len(clean))
clean.head()
```

    rows before: 150  after: 146
    




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



## One-hot mapping of a categorical attribute


```python
cat_data = pd.DataFrame({"color": ["red", "blue", "green", "red"], "value": [10, 20, 15, 12]})
dal.CategMapping("color").fit(cat_data).transform(cat_data)
```




       value  color_red  color_blue  color_green
    0     10          1           0            0
    1     20          0           1            0
    2     15          0           0            1
    3     12          1           0            0



Many beginners jump straight to modeling; in practice, data-quality decisions shape the
experiment before the learner ever sees the data.

# Categorical Mapping — One-Hot (CategMapping)

## About the transformation
- `CategMapping`: converts a categorical column into binary (one-hot) variables, one output column per category.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Apply one-hot mapping to the `Species` column


```python
iris = dal.datasets.iris()
cm = dal.CategMapping("Species").fit(iris)
cm.transform(iris).head()
```




       sepal_length  sepal_width  petal_length  petal_width  Species_setosa  Species_versicolor  Species_virginica
    0           5.1          3.5           1.4          0.2               1                   0                  0
    1           4.9          3.0           1.4          0.2               1                   0                  0
    2           4.7          3.2           1.3          0.2               1                   0                  0
    3           4.6          3.1           1.5          0.2               1                   0                  0
    4           5.0          3.6           1.4          0.2               1                   0                  0



## References
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer. (1-of-K / one-hot encoding)

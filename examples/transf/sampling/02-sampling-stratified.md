# Stratified Sampling (SampleStratified)

## About sampling
- `SampleStratified`: splits train/test and folds preserving the target variable proportion (stratification) per category.

## Didactic goal
Focus on what changes in the dataset after each operation. Understanding the effect of the transformation is as important as learning the function name.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load dataset and view class distribution


```python
iris = dal.datasets.iris()
iris["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    50
    virginica     50
    Name: count, dtype: int64



## Stratified train/test split preserving `Species` proportions


```python
tt = dal.SampleStratified("Species", seed=1).train_test(iris)
pd.DataFrame({
    "training": tt["train"]["Species"].value_counts(),
    "test": tt["test"]["Species"].value_counts(),
})
```




                training  test
    Species                   
    setosa            40    10
    versicolor        40    10
    virginica         40    10



## Create stratified folds and check distribution


```python
folds = dal.SampleStratified("Species", seed=1).k_fold(iris, 4)
pd.DataFrame([f["Species"].value_counts() for f in folds])
```




    Species  setosa  virginica  versicolor
    count        13         13          12
    count        13         13          12
    count        12         12          13
    count        12         12          13



## References
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI.

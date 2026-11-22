# Tutorial 02 - Sampling Strategy

A good experiment starts with a good split. When the target is categorical, stratified sampling
keeps class proportions similar across subsets, so evaluation is not distorted by the partition.

## Original class distribution


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
iris["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    50
    virginica     50
    Name: count, dtype: int64



## Stratified train/test split


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



## Stratified folds for cross-validation style workflows


```python
folds = dal.SampleStratified("Species", seed=1).k_fold(iris, 4)
pd.DataFrame([f["Species"].value_counts() for f in folds])
```




    Species  setosa  virginica  versicolor
    count        13         13          12
    count        13         13          12
    count        12         12          13
    count        12         12          13



## A baseline on the stratified split


```python
slevels = list(iris["Species"].cat.categories)
model = dal.ClaMajority("Species", slevels).fit(tt["train"])
pred = model.predict(tt["test"].drop(columns="Species"))
model.evaluate(tt["test"]["Species"], pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.333333  10   0  20   0   0.333333     1.0          1.0          0.0  0.5



Before comparing models, make sure the data partition itself is not introducing distortion.

# Gradient Boosting — XGBoost (ClaXgboost)

## About the method
- `ClaXgboost`: gradient boosting classifier using the `xgboost` backend (optional extra `[xgboost]`).

## Didactic goal
Keep the multiclass classification line of experiment fixed and show how a more configurable boosting engine still fits the same DAL workflow.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data and inspect.


```python
iris = dal.datasets.iris()
iris.head()
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



## Target `Species` levels and reproducible train/test split.


```python
slevels = list(iris["Species"].cat.categories)
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## Class distribution by split


```python
pd.DataFrame({
    "dataset": iris["Species"].value_counts(),
    "training": train["Species"].value_counts(),
    "test": test["Species"].value_counts(),
}).T
```




    Species   setosa  versicolor  virginica
    dataset       50          50         50
    training      39          40         41
    test          11          10          9



## Model configuration and fitting


```python
model = dal.ClaXgboost("Species", slevels, nrounds=5, random_state=1).fit(train)
```

## Training evaluation


```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0     0.975  39  81   0   0        1.0     1.0          1.0          1.0  1.0



## Test evaluation


```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity        f1
    0  0.966667  11  18   1   0   0.916667     1.0          1.0     0.947368  0.956522



## What to observe
- The experiment body is the same as in the other multiclass classification examples.
- The method-specific change is in the boosting engine and its configuration, not in the workflow.

## References
- Chen, T., and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*

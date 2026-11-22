# Regularized Logistic Regression (ClaGlmnet)

## About the method
- `ClaGlmnet`: L1-regularized (lasso) logistic regression for binary classification.

## Didactic goal
Keep the same binary classification line of experiment used for `ClaGlm` and change only the learner and its regularization choice.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data, derive a binary target, and inspect.


```python
iris = dal.datasets.iris().copy()
iris["IsVersicolor"] = (iris["Species"] == "versicolor").map(
    {True: "versicolor", False: "not_versicolor"}
).astype("category")
iris = iris.drop(columns="Species")
iris.head()
```




       sepal_length  sepal_width  petal_length  petal_width    IsVersicolor
    0           5.1          3.5           1.4          0.2  not_versicolor
    1           4.9          3.0           1.4          0.2  not_versicolor
    2           4.7          3.2           1.3          0.2  not_versicolor
    3           4.6          3.1           1.5          0.2  not_versicolor
    4           5.0          3.6           1.4          0.2  not_versicolor



## Target levels and reproducible stratified split.


```python
slevels = list(iris["IsVersicolor"].cat.categories)
split = dal.SampleStratified("IsVersicolor", seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## Class distribution by split


```python
pd.DataFrame({
    "dataset": iris["IsVersicolor"].value_counts(),
    "training": train["IsVersicolor"].value_counts(),
    "test": test["IsVersicolor"].value_counts(),
}).T
```




    IsVersicolor  not_versicolor  versicolor
    dataset                  100          50
    training                  80          40
    test                      20          10



## Model configuration and fitting


```python
model = dal.ClaGlmnet("IsVersicolor", slevels, alpha=1.0, random_state=1).fit(train)
```

## Training evaluation


```python
train_pred = model.predict(train.drop(columns="IsVersicolor"))
model.evaluate(train["IsVersicolor"], train_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity        f1
    0      0.75  70  20  20  10   0.777778   0.875        0.875          0.5  0.823529



## Test evaluation


```python
test_pred = model.predict(test.drop(columns="IsVersicolor"))
model.evaluate(test["IsVersicolor"], test_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0       0.7  18   3   7   2       0.72     0.9          0.9          0.3  0.8



## What to observe
- The binary experiment structure is unchanged from `ClaGlm`.
- The key method-specific choice is how the penalized solution is selected through the regularization strength.

## References
- Friedman, J., Hastie, T., and Tibshirani, R. (2010). *Regularization Paths for Generalized Linear Models via Coordinate Descent.*

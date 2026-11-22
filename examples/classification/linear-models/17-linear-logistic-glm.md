# Logistic Regression (ClaGlm)

## About the method
- `ClaGlm`: logistic regression using the generalized linear model framework.

## Didactic goal
Preserve the same line of experiment and change only what the method requires. Here the special requirement is a binary target instead of the original three-class `Species` problem.

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
model = dal.ClaGlm("IsVersicolor", slevels, random_state=1).fit(train)
```

## Training evaluation


```python
train_pred = model.predict(train.drop(columns="IsVersicolor"))
model.evaluate(train["IsVersicolor"], train_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity        f1
    0  0.733333  70  18  22  10    0.76087   0.875        0.875         0.45  0.813953



## Test evaluation


```python
test_pred = model.predict(test.drop(columns="IsVersicolor"))
model.evaluate(test["IsVersicolor"], test_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity        f1
    0       0.7  19   2   8   1   0.703704    0.95         0.95          0.2  0.808511



## What to observe
- The experiment body is the same as in multiclass classification.
- The only structural adaptation is the binary target required by logistic regression.

## References
- McCullagh, P., and Nelder, J. A. (1989). *Generalized Linear Models.*

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---



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

## Model configuration and fitting
```python
model = dal.ClaGlmnet("IsVersicolor", slevels, alpha=1.0, random_state=1).fit(train)
```

## Training evaluation
```python
train_pred = model.predict(train.drop(columns="IsVersicolor"))
model.evaluate(train["IsVersicolor"], train_pred)["metrics"]
```

## Test evaluation
```python
test_pred = model.predict(test.drop(columns="IsVersicolor"))
model.evaluate(test["IsVersicolor"], test_pred)["metrics"]
```

## What to observe
- The binary experiment structure is unchanged from `ClaGlm`.
- The key method-specific choice is how the penalized solution is selected through the regularization strength.

## References
- Friedman, J., Hastie, T., and Tibshirani, R. (2010). *Regularization Paths for Generalized Linear Models via Coordinate Descent.*

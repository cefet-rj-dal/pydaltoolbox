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
model = dal.ClaGlm("IsVersicolor", slevels, random_state=1).fit(train)
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
- The experiment body is the same as in multiclass classification.
- The only structural adaptation is the binary target required by logistic regression.

## References
- McCullagh, P., and Nelder, J. A. (1989). *Generalized Linear Models.*

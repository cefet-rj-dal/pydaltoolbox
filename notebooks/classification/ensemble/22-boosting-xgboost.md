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

## Model configuration and fitting
```python
model = dal.ClaXgboost("Species", slevels, nrounds=5, random_state=1).fit(train)
```

## Training evaluation
```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```

## Test evaluation
```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```

## What to observe
- The experiment body is the same as in the other multiclass classification examples.
- The method-specific change is in the boosting engine and its configuration, not in the workflow.

## References
- Chen, T., and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*

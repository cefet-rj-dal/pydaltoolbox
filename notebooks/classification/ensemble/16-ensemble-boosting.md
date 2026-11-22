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



# Boosting (ClaBoosting)


## About the method
- `ClaBoosting`: boosting classifier that builds a sequence of weak learners, each one focusing more on previously difficult cases.

## Didactic goal
Keep the line of experiment fixed and isolate the change in modeling idea. This helps the reader compare boosting with bagging and random forests without changing the data or evaluation structure.

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
model = dal.ClaBoosting("Species", slevels, mfinal=20, random_state=1).fit(train)
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
- Boosting changes the learner sequentially, not only by averaging independent resamples.
- The experiment body remains the same even though the ensemble logic differs from bagging and forests.

## References
- Freund, Y., and Schapire, R. E. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.*

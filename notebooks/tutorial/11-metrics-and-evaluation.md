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

# Tutorial 11 - Metrics and Evaluation

Running `evaluate` is easy; interpreting the result is where learning starts. Training and test
metrics should be read together.

## Train a decision tree on a standard split
```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]

model = dal.ClaDtree("Species", slevels, random_state=1).fit(train)
```

## Training metrics (usually optimistic)
```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```

## Test metrics (the important number for generalization)
```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```

## Confusion matrix
```python
predicted = test_pred.idxmax(axis=1)
pd.crosstab(predicted, test["Species"], rownames=["predicted"], colnames=["observed"])
```

Rule of thumb: if training results are much better than test results, the model may be too
adapted to the training data.

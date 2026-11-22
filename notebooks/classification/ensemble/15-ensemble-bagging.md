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



# Bagging (ClaBagging)


## About the method
- `ClaBagging`: bootstrap aggregation of classification trees. The main idea is to reduce variance by fitting many trees on bootstrap samples and combining their votes.

## Didactic goal
Keep the same classification line of experiment used in the earlier examples and change only the learner family. This makes it easier to compare what bagging changes relative to a single-tree baseline.

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
model = dal.ClaBagging("Species", slevels, nbagg=25, random_state=1).fit(train)
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
- Bagging keeps the same DAL workflow while changing the model-construction logic internally.
- The learner usually becomes more stable than a single tree because the prediction is aggregated across bootstrap models.

## References
- Breiman, L. (1996). *Bagging Predictors.*

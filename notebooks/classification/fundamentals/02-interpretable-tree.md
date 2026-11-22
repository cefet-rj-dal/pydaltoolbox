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



# Decision Tree (ClaDtree)


## About the method
- `ClaDtree`: decision tree for classification. Recursively splits explanatory variables to separate classes, yielding an interpretable model.

## Didactic goal
Keep exactly the same classification line of experiment introduced by the baseline and change only the learner to an interpretable supervised model.

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
model = dal.ClaDtree("Species", slevels, random_state=1).fit(train)
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
- The experiment structure is unchanged from the baseline example.
- The difference now is that the learner can express feature-based decision rules instead of only the majority class.

## References
- Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). *Classification and Regression Trees.*

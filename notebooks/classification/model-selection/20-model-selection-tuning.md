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



# Model Selection — Tuning (ClaTune)


## About the method
- `ClaTune`: hyperparameter search for a classifier over ranges defined in `ranges`, by k-fold cross-validation; refits the best configuration.
- In this example the tuned base learner is `ClaSvm`.

## Didactic goal
Preserve the same classification line of experiment and treat tuning as a change in model configuration rather than as a separate workflow.

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
tune = dal.ClaTune(
    dal.ClaSvm("Species", slevels),
    folds=5,
    ranges={
        "cost": [20, 40, 60, 80, 100],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
    },
    seed=42,
)
model = tune.fit(train)
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
- The experiment body is still split, fit, predict, and evaluate.
- Tuning changes how the model is selected, not how the experiment is read.

## References
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.*

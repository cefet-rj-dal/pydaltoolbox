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



# Support Vector Regression (RegSvm)


## About the method
- `RegSvm`: Support Vector Regression.
- Main parameters: `epsilon`, `cost`, and `kernel`.

## Didactic goal
Keep the same regression line of experiment and change only the learner to a margin-based regressor.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data and inspect
```python
data = dal.datasets.diabetes()
data.head()
```

## Target `target` and reproducible train/test split
```python
split = dal.SampleRandom(seed=1).train_test(data)
train, test = split["train"], split["test"]
```

## Model configuration and fitting
```python
model = dal.RegSvm("target", epsilon=0.2, cost=40.0).fit(train)
```

## Training evaluation
```python
train_pred = model.predict(train.drop(columns="target"))
model.evaluate(train["target"], train_pred)["metrics"]
```

## Test evaluation
```python
test_pred = model.predict(test.drop(columns="target"))
model.evaluate(test["target"], test_pred)["metrics"]
```

## References
- Drucker, H., Burges, C., Kaufman, L., Smola, A., and Vapnik, V. (1997). *Support Vector Regression Machines.*

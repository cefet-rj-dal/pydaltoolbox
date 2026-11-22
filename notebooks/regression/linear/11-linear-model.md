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



# Linear Regression (RegLm)


## About the method
- `RegLm`: ordinary linear regression model.

## Didactic goal
Preserve the same regression line of experiment and use a linear model as a reference point before comparing nonlinear or ensemble regressors.

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
model = dal.RegLm("target").fit(train)
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

## What to observe
- The regression workflow is the same as in the other examples.
- The method-specific difference is the linear relationship assumed between predictors and target.

## References
- Montgomery, D. C., Peck, E. A., and Vining, G. G. (2021). *Introduction to Linear Regression Analysis.*

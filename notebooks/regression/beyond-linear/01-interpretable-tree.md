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



# Regression Tree (RegDtree)


## About the method
- `RegDtree`: decision tree for regression. Partitions feature space and predicts numeric values by leaf means.

## Didactic goal
Establish the standard regression line of experiment used throughout this family of examples.

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
model = dal.RegDtree("target", random_state=1).fit(train)
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
- The regression workflow already has the same structural rhythm as classification.
- What changes is the target type and the interpretation of the evaluation metrics.

## References
- Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). *Classification and Regression Trees.*

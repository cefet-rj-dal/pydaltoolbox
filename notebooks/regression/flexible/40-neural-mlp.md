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



# Multilayer Perceptron Regression (RegMlp)


## About the method
- `RegMlp`: Multilayer Perceptron for regression.
- Main parameters: `size` and `decay`.

## Didactic goal
Keep the same regression line of experiment and change only the learner to a neural regressor.

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
model = dal.RegMlp("target", size=5, decay=0.54, random_state=1).fit(train)
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
- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition.*

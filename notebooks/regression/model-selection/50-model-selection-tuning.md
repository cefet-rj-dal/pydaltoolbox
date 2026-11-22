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



# Model Selection — Tuning (RegTune)


## About the method
- `RegTune`: hyperparameter search for regression models over ranges defined in `ranges`, by k-fold cross-validation.
- In this example the tuned base learner is `RegSvm`.

## Didactic goal
Preserve the same regression line of experiment and treat tuning as a change in model configuration rather than as a separate workflow.

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
tune = dal.RegTune(
    dal.RegSvm("target"),
    folds=5,
    ranges={
        "epsilon": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "cost": [20, 40, 60, 80, 100],
        "kernel": ["rbf"],
    },
    seed=42,
)
model = tune.fit(train)
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
- The workflow is still split, fit, predict, and evaluate.
- Tuning changes how the regressor is chosen, not how the line of experiment is structured.

## References
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.*

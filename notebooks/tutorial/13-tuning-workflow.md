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

# Tutorial 13 - Tuning Workflow

After comparing default learners, ask whether one can be improved by hyperparameter search.
Tuning is a disciplined extension of the same workflow, not a separate world.

## Prepare the classification problem
```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## Define a compact search space and fit the tuned learner
```python
tune = dal.ClaTune(
    dal.ClaSvm("Species", slevels),
    folds=5,
    ranges={
        "cost": [20, 40, 60],
        "kernel": ["linear", "rbf"],
    },
    seed=42,
)
model = tune.fit(train)
```

## Evaluate on training and test data
```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```

```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```

Tuning should refine a solid workflow, not replace careful sampling and evaluation.

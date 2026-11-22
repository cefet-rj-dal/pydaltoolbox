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

# Tutorial 21 - Regression Workflow

The Experiment Line is not limited to classification. In regression the target is numeric, but
the stages stay the same: split, fit, predict, evaluate.

## Load a numeric prediction problem (predict `target`)
```python
import pydaltoolbox as dal

diabetes = dal.datasets.diabetes()
diabetes.head()
```

## Train/test split
```python
split = dal.SampleRandom(seed=1).train_test(diabetes)
train, test = split["train"], split["test"]
```

## Fit a regression tree; inspect training and test errors
```python
model = dal.RegDtree("target", random_state=1).fit(train)

train_pred = model.predict(train.drop(columns="target"))
print(model.evaluate(train["target"], train_pred)["metrics"])

test_pred = model.predict(test.drop(columns="target"))
model.evaluate(test["target"], test_pred)["metrics"]
```

The workflow is familiar, but the metrics now belong to regression. The same
`fit -> predict -> evaluate` contract is what lets `RegLm` and `RegTune` integrate into the
same Experiment Line.

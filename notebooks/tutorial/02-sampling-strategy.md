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

# Tutorial 02 - Sampling Strategy

A good experiment starts with a good split. When the target is categorical, stratified sampling
keeps class proportions similar across subsets, so evaluation is not distorted by the partition.

## Original class distribution
```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
iris["Species"].value_counts()
```

## Stratified train/test split
```python
tt = dal.SampleStratified("Species", seed=1).train_test(iris)
pd.DataFrame({
    "training": tt["train"]["Species"].value_counts(),
    "test": tt["test"]["Species"].value_counts(),
})
```

## Stratified folds for cross-validation style workflows
```python
folds = dal.SampleStratified("Species", seed=1).k_fold(iris, 4)
pd.DataFrame([f["Species"].value_counts() for f in folds])
```

## A baseline on the stratified split
```python
slevels = list(iris["Species"].cat.categories)
model = dal.ClaMajority("Species", slevels).fit(tt["train"])
pred = model.predict(tt["test"].drop(columns="Species"))
model.evaluate(tt["test"]["Species"], pred)["metrics"]
```

Before comparing models, make sure the data partition itself is not introducing distortion.

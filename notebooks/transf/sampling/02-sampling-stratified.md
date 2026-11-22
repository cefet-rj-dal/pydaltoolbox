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

# Stratified Sampling (SampleStratified)

## About sampling
- `SampleStratified`: splits train/test and folds preserving the target variable proportion (stratification) per category.

## Didactic goal
Focus on what changes in the dataset after each operation. Understanding the effect of the transformation is as important as learning the function name.

## Environment setup
```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load dataset and view class distribution
```python
iris = dal.datasets.iris()
iris["Species"].value_counts()
```

## Stratified train/test split preserving `Species` proportions
```python
tt = dal.SampleStratified("Species", seed=1).train_test(iris)
pd.DataFrame({
    "training": tt["train"]["Species"].value_counts(),
    "test": tt["test"]["Species"].value_counts(),
})
```

## Create stratified folds and check distribution
```python
folds = dal.SampleStratified("Species", seed=1).k_fold(iris, 4)
pd.DataFrame([f["Species"].value_counts() for f in folds])
```

## References
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI.

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

# Aggregation by Group (Aggregation)

## About the transformation
- `Aggregation`: summarizes a dataset by groups through named aggregation expressions.

## Didactic goal
Show that some transformations are not instance-level edits but dataset-level restructuring steps. Aggregation is useful before plotting, reporting, and grouped comparison.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Summarize sepal length by species
```python
iris = dal.datasets.iris()
agg = dal.Aggregation(
    "Species",
    mean_sepal=("sepal_length", "mean"),
    sd_sepal=("sepal_length", "std"),
    n=("sepal_length", "size"),
)
agg.transform(iris)
```

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

# Feature Generation (FeatureGeneration)

## About the transformation
- `FeatureGeneration`: creates new attributes from named expressions over existing columns.

## Didactic goal
Reinforce that preprocessing is not only about cleaning or scaling. Feature generation changes the representation itself and can make patterns easier for downstream learners to capture.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Create area and ratio features
```python
iris = dal.datasets.iris()
gen = dal.FeatureGeneration(
    sepal_area=lambda df: df["sepal_length"] * df["sepal_width"],
    petal_area=lambda df: df["petal_length"] * df["petal_width"],
    sepal_ratio="sepal_length / sepal_width",
)
gen.transform(iris).head()
```

## What to observe
- Generated attributes become regular columns in the transformed dataset.
- This is often one of the most impactful preprocessing steps in practice.

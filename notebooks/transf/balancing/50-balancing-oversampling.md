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

# Class Oversampling — Random or SMOTE (BalOversampling)

## About the transformation
This example balances minority classes either by random replication or by synthetic interpolation (SMOTE).

## Didactic goal
Compare two ways of increasing minority-class representation without changing the modeling interface. The balancing strategy changes the type of synthetic information introduced into the training data.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Create an imbalanced version of iris
```python
iris = dal.datasets.iris()
imbalanced = iris.iloc[list(range(0, 50)) + list(range(50, 71)) + list(range(100, 111))]
imbalanced["Species"].value_counts()
```

## Random oversampling duplicates existing minority cases
```python
dal.BalOversampling("Species", method="random", seed=1).transform(imbalanced)["Species"].value_counts()
```

## SMOTE creates synthetic minority instances by interpolation
```python
dal.BalOversampling("Species", method="smote", k=3, seed=1).transform(imbalanced)["Species"].value_counts()
```

## What to observe
- Random oversampling preserves original minority examples exactly (duplicated rows may appear).
- SMOTE changes the dataset more deeply because it generates synthetic numeric combinations.
- Both should be fitted only on the training split, to avoid leakage into evaluation.

## References
- Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.*

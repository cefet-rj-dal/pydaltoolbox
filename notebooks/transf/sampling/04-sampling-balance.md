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

# Class Balancing via Sampling (SampleBalance)

## About the transformation
- `SampleBalance`: class balancing by up-sampling or down-sampling through the sampling interface.

## Didactic goal
Compare this general balancing helper with the more explicit oversampling and subsampling examples. The emphasis is on class-count correction, not on synthetic data generation.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Create an imbalanced subset and then rebalance it
```python
iris = dal.datasets.iris()
imbalanced = iris.iloc[list(range(0, 50)) + list(range(50, 71)) + list(range(100, 111))]
imbalanced["Species"].value_counts()
```

## Down-sampling reduces every class to the minority count
```python
dal.SampleBalance("Species", method="down", seed=1).transform(imbalanced)["Species"].value_counts()
```

## Up-sampling replicates minority classes up to the majority count
```python
dal.SampleBalance("Species", method="up", seed=1).transform(imbalanced)["Species"].value_counts()
```

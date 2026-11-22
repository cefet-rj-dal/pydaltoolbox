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

# Simple Sampling (SampleSimple)

## About the transformation
- `SampleSimple`: simple random sampling over rows, with or without replacement.

## Didactic goal
Distinguish basic sampling of records from train/test partitioning. This object is useful when the task is to extract a subset, not to organize an experimental protocol.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Draw one sample without replacement and another with replacement
```python
iris = dal.datasets.iris()
sample_wor = dal.SampleSimple(size=10, seed=1).transform(iris)
sample_wr = dal.SampleSimple(size=10, replace=True, seed=1).transform(iris)
sample_wor["sepal_length"].tolist()
```

## The with-replacement draw may repeat rows
```python
sample_wr["sepal_length"].tolist()
```

## References
- Cochran, W. G. (1977). *Sampling Techniques.*

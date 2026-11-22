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

# Min-Max Normalization (MinMax)

## About the transformation
- `MinMax`: linearly rescales numeric attributes to [0, 1]. Useful for scale-sensitive algorithms and models that expect bounded inputs.
- For each numeric column: `(x - min) / (max - min)`; constant columns map to 0.

## Didactic goal
The value of the example is not only the call to `MinMax()`, but the before/after comparison. This is one of the most common steps before distance-based or scale-sensitive learners.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Context and sample data
```python
iris = dal.datasets.iris()
iris.describe()
```

## Apply Min-Max and inspect the resulting scale
```python
norm = dal.MinMax().fit(iris)
ndata = norm.transform(iris)
ndata.describe()
```

## Inverse transform (denormalize) to verify integrity
```python
norm.inverse_transform(ndata).describe()
```

## What to observe
- Numeric attributes that originally had different ranges are now on a common scale.
- The categorical target remains untouched, which is usually what you want in supervised tasks.

## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Normalization)

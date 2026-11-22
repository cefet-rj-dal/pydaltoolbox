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

# Minimum Curvature / Knee (FitCurvatureMin)

## About the technique
- `FitCurvatureMin`: computes curvature via the second derivative of a smoothed spline and returns the minimum-curvature position (knee) for increasing curves; useful to find where additional gains become marginal.

## Environment setup
```python
import warnings

import numpy as np

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Build a PCA cumulative-variance curve (an increasing curve)
```python
x = dal.datasets.iris().drop(columns="Species").to_numpy()
xs = (x - x.mean(axis=0)) / x.std(axis=0)
eig = np.sort(np.linalg.eigvalsh(np.cov(xs, rowvar=False)))[::-1]
cumvar = np.cumsum(eig) / eig.sum()
cumvar
```

## Find the knee (minimum curvature)
```python
res = dal.FitCurvatureMin().transform(cumvar)
res
```

## References
- Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). *Finding a "Kneedle" in a Haystack.*

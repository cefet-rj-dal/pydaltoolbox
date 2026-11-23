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




    array([0.72962445, 0.95813207, 0.99482129, 1.        ])



## Find the knee (minimum curvature)


```python
res = dal.FitCurvatureMin().transform(cumvar)
res
```




       x         y          yfit
    0  2  0.958132 -3.114198e-09



## References
- Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). *Finding a "Kneedle" in a Haystack.*

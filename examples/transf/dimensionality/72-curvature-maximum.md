# Maximum Curvature / Knee (FitCurvatureMax)

## About the technique
- `FitCurvatureMax`: computes curvature via the second derivative of a smoothed spline and returns the maximum-curvature position for decreasing curves; useful to choose where further reductions add little benefit.

## Environment setup


```python
import warnings

import numpy as np

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Synthetic decreasing curve (-log) to illustrate the maximum-curvature point


```python
x = np.arange(1, 10.5, 0.5)
y = -np.log(x)
res = dal.FitCurvatureMax().transform(y)
res
```




       x         y     yfit
    0  2 -0.405465  0.16553



## References
- Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). *Finding a "Kneedle" in a Haystack.*

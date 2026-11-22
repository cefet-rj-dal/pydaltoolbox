# Feature Selection with Lasso (FeatureSelectionLasso)

## About the transformation
Lasso performs regression with an L1 penalty, shrinking some coefficients exactly to zero. This example keeps only the predictors with non-zero coefficients at the selected regularization level.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Lasso selection on the diabetes regression target


```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionLasso("target").fit(diabetes)
fs.transform(diabetes).head()
```




            age       sex       bmi        bp        s1        s2        s4        s5        s6  target
    0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.002592  0.019907 -0.017646   151.0
    1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163 -0.039493 -0.068332 -0.092204    75.0
    2  0.085299  0.050680  0.044451 -0.005670 -0.045599 -0.034194 -0.002592  0.002861 -0.025930   141.0
    3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991  0.034309  0.022688 -0.009362   206.0
    4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596 -0.002592 -0.031988 -0.046641   135.0



## References
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.*

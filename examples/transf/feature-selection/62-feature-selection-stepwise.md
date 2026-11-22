# Feature Selection with Stepwise Search (FeatureSelectionStepwise)

## About the transformation
Stepwise search iteratively adds or removes predictors from a generalized linear model according to an information criterion. This example uses forward search for a regression target.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Forward stepwise selection on the diabetes regression target


```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionStepwise("target", family="gaussian", direction="forward").fit(diabetes)
fs.transform(diabetes).head()
```




            sex       bmi        bp        s1        s2        s5  target
    0  0.050680  0.061696  0.021872 -0.044223 -0.034821  0.019907   151.0
    1 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163 -0.068332    75.0
    2  0.050680  0.044451 -0.005670 -0.045599 -0.034194  0.002861   141.0
    3 -0.044642 -0.011595 -0.036656  0.012191  0.024991  0.022688   206.0
    4 -0.044642 -0.036385  0.021872  0.003935  0.015596 -0.031988   135.0



## References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning.*

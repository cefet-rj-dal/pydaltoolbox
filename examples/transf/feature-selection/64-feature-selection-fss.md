# Forward Stepwise Subset Search (FeatureSelectionFss)

## About the transformation
Forward stepwise subset search starts with no predictors and adds the feature that most improves the current regression model, keeping the subset with the best score.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Forward subset selection on the diabetes regression target


```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionFss("target").fit(diabetes)
fs.transform(diabetes).head()
```




            sex       bmi        bp        s1        s2        s4        s5        s6  target
    0  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.002592  0.019907 -0.017646   151.0
    1 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163 -0.039493 -0.068332 -0.092204    75.0
    2  0.050680  0.044451 -0.005670 -0.045599 -0.034194 -0.002592  0.002861 -0.025930   141.0
    3 -0.044642 -0.011595 -0.036656  0.012191  0.024991  0.034309  0.022688 -0.009362   206.0
    4 -0.044642 -0.036385  0.021872  0.003935  0.015596 -0.002592 -0.031988 -0.046641   135.0



## References
- Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning.*

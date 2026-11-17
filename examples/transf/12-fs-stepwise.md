# Feature Selection by Forward Stepwise (FeatureSelectionStepwise)

Forward stepwise search starts from no predictors and adds, one at a time, the feature
that most improves the fit; it then keeps the subset size with the best adjusted
R-squared (which penalizes adding uninformative predictors). The target must be
numeric.

**Reference:** Hocking, R. R. (1976). *The Analysis and Selection of Variables in
Linear Regression.* Biometrics.


```python
import pydaltoolbox as dal

diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionStepwise("target").fit(diabetes)
print("selected:", fs.selected)
```

    selected: ['bmi', 's5', 'bp', 's1', 'sex', 's2', 's4', 's6']
    


```python
print("features before:", diabetes.shape[1] - 1, " after:", len(fs.selected))
fs.transform(diabetes).head()
```

    features before: 10  after: 8
    




            sex       bmi        bp        s1        s2        s4        s5        s6  target
    0  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.002592  0.019907 -0.017646   151.0
    1 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163 -0.039493 -0.068332 -0.092204    75.0
    2  0.050680  0.044451 -0.005670 -0.045599 -0.034194 -0.002592  0.002861 -0.025930   141.0
    3 -0.044642 -0.011595 -0.036656  0.012191  0.024991  0.034309  0.022688 -0.009362   206.0
    4 -0.044642 -0.036385  0.021872  0.003935  0.015596 -0.002592 -0.031988 -0.046641   135.0



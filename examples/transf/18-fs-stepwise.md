# Feature Selection by Stepwise GLM (FeatureSelectionStepwise)

Stepwise selection fits a generalized linear model and adds and/or drops predictors one
at a time, each step taking the move that most lowers Akaike's Information Criterion
(`AIC = 2k - 2*logLik`). AIC rewards fit while penalizing model size. This mirrors the R
`feature_selection_stepwise` (`stats::step`): `direction` is `"forward"`, `"backward"`,
or `"both"`, and `family` is `"binomial"` (default, logistic — binary target) or
`"gaussian"` (linear — numeric target).

**Reference:** Akaike, H. (1974). *A New Look at the Statistical Model Identification.*
IEEE Transactions on Automatic Control.

## Binary target (binomial, default)


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris = iris.assign(is_versicolor=(iris["Species"] == "versicolor").astype(int))
iris = iris.drop(columns=["Species"])

fs = dal.FeatureSelectionStepwise("is_versicolor", direction="forward").fit(iris)
print("selected:", fs.selected)
fs.transform(iris).head()
```

    selected: ['sepal_width']
    




       sepal_width  is_versicolor
    0          3.5              0
    1          3.0              0
    2          3.2              0
    3          3.1              0
    4          3.6              0



## Numeric target (gaussian)


```python
diabetes = dal.datasets.diabetes()
fs = dal.FeatureSelectionStepwise("target", family="gaussian", direction="both").fit(diabetes)
print("selected:", fs.selected)
fs.transform(diabetes).head()
```

    selected: ['sex', 'bmi', 'bp', 's1', 's2', 's5']
    




            sex       bmi        bp        s1        s2        s5  target
    0  0.050680  0.061696  0.021872 -0.044223 -0.034821  0.019907   151.0
    1 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163 -0.068332    75.0
    2  0.050680  0.044451 -0.005670 -0.045599 -0.034194  0.002861   141.0
    3 -0.044642 -0.011595 -0.036656  0.012191  0.024991  0.022688   206.0
    4 -0.044642 -0.036385  0.021872  0.003935  0.015596 -0.031988   135.0



# Linear Regression (RegLm)

## About the method
- `RegLm`: ordinary linear regression model.

## Didactic goal
Preserve the same regression line of experiment and use a linear model as a reference point before comparing nonlinear or ensemble regressors.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data and inspect


```python
data = dal.datasets.diabetes()
data.head()
```




            age       sex       bmi        bp        s1        s2        s3        s4        s5        s6  target
    0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019907 -0.017646   151.0
    1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068332 -0.092204    75.0
    2  0.085299  0.050680  0.044451 -0.005670 -0.045599 -0.034194 -0.032356 -0.002592  0.002861 -0.025930   141.0
    3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022688 -0.009362   206.0
    4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031988 -0.046641   135.0



## Target `target` and reproducible train/test split


```python
split = dal.SampleRandom(seed=1).train_test(data)
train, test = split["train"], split["test"]
```

## Model configuration and fitting


```python
model = dal.RegLm("target").fit(train)
```

## Training evaluation


```python
train_pred = model.predict(train.drop(columns="target"))
model.evaluate(train["target"], train_pred)["metrics"]
```




               mse     smape        R2
    0  2824.560082  0.309123  0.533499



## Test evaluation


```python
test_pred = model.predict(test.drop(columns="target"))
model.evaluate(test["target"], test_pred)["metrics"]
```




               mse    smape        R2
    0  3079.586646  0.33559  0.432658



## What to observe
- The regression workflow is the same as in the other examples.
- The method-specific difference is the linear relationship assumed between predictors and target.

## References
- Montgomery, D. C., Peck, E. A., and Vining, G. G. (2021). *Introduction to Linear Regression Analysis.*

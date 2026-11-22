# Regression Tree (RegDtree)

## About the method
- `RegDtree`: decision tree for regression. Partitions feature space and predicts numeric values by leaf means.

## Didactic goal
Establish the standard regression line of experiment used throughout this family of examples.

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
model = dal.RegDtree("target", random_state=1).fit(train)
```

## Training evaluation


```python
train_pred = model.predict(train.drop(columns="target"))
model.evaluate(train["target"], train_pred)["metrics"]
```




               mse     smape        R2
    0  1340.975483  0.204274  0.778526



## Test evaluation


```python
test_pred = model.predict(test.drop(columns="target"))
model.evaluate(test["target"], test_pred)["metrics"]
```




               mse     smape        R2
    0  5444.302157  0.432624 -0.002986



## What to observe
- The regression workflow already has the same structural rhythm as classification.
- What changes is the target type and the interpretation of the evaluation metrics.

## References
- Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). *Classification and Regression Trees.*

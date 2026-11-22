# Tutorial 21 - Regression Workflow

The Experiment Line is not limited to classification. In regression the target is numeric, but
the stages stay the same: split, fit, predict, evaluate.

## Load a numeric prediction problem (predict `target`)


```python
import pydaltoolbox as dal

diabetes = dal.datasets.diabetes()
diabetes.head()
```




            age       sex       bmi        bp        s1        s2        s3        s4        s5        s6  target
    0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019907 -0.017646   151.0
    1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068332 -0.092204    75.0
    2  0.085299  0.050680  0.044451 -0.005670 -0.045599 -0.034194 -0.032356 -0.002592  0.002861 -0.025930   141.0
    3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022688 -0.009362   206.0
    4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031988 -0.046641   135.0



## Train/test split


```python
split = dal.SampleRandom(seed=1).train_test(diabetes)
train, test = split["train"], split["test"]
```

## Fit a regression tree; inspect training and test errors


```python
model = dal.RegDtree("target", random_state=1).fit(train)

train_pred = model.predict(train.drop(columns="target"))
print(model.evaluate(train["target"], train_pred)["metrics"])

test_pred = model.predict(test.drop(columns="target"))
model.evaluate(test["target"], test_pred)["metrics"]
```

               mse     smape        R2
    0  1340.975483  0.204274  0.778526
    




               mse     smape        R2
    0  5444.302157  0.432624 -0.002986



The workflow is familiar, but the metrics now belong to regression. The same
`fit -> predict -> evaluate` contract is what lets `RegLm` and `RegTune` integrate into the
same Experiment Line.

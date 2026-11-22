# Custom Regression

The same customization idea as custom classification, but for a numeric target: implement
`fit` and `predict` over an external backend and reuse the regression Experiment Line. Here
the custom regressor wraps scikit-learn's gradient boosting and runs on the diabetes dataset.

## A custom regressor subclassing `Regression`


```python
from sklearn.ensemble import GradientBoostingRegressor

import pydaltoolbox as dal


class RegGbrCustom(dal.Regression):
    def __init__(self, attribute, n_estimators=100):
        super().__init__(attribute)
        self.n_estimators = n_estimators

    def fit(self, data, **kwargs):
        x = data.drop(columns=self.attribute)
        y = data[self.attribute]
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators, random_state=1
        ).fit(x, y)
        return self

    def predict(self, x, **kwargs):
        return self.model.predict(x)
```

## Same regression workflow


```python
data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=1).train_test(data)
train, test = split["train"], split["test"]

model = RegGbrCustom("target", n_estimators=100).fit(train)
```

## Training and test evaluation


```python
train_pred = model.predict(train.drop(columns="target"))
model.evaluate(train["target"], train_pred)["metrics"]
```




              mse     smape        R2
    0  908.401583  0.188039  0.849969




```python
test_pred = model.predict(test.drop(columns="target"))
model.evaluate(test["target"], test_pred)["metrics"]
```




              mse     smape       R2
    0  3556.81506  0.341703  0.34474



## References
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*

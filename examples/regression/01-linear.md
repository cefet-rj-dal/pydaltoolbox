# Linear Regression (RegLm)

Linear regression models the target as a linear combination of the features, fit by
ordinary least squares. It is the canonical interpretable baseline for numeric
prediction: coefficients show each feature's direction and magnitude, and it trains
instantly. `RegLm` wraps scikit-learn's `LinearRegression`.

Regression is evaluated with MSE (mean squared error), sMAPE (symmetric mean absolute
percentage error) and R2 (coefficient of determination).

**Reference:** Montgomery, D. C., Peck, E. A., Vining, G. G. (2012). *Introduction to
Linear Regression Analysis.* Wiley.


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
```


```python
model = dal.RegLm("target").fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["target"], model.predict(test.drop(columns="target")))["metrics"]
```




               mse     smape        R2
    0  2594.629706  0.295464  0.592695



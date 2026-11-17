---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Random Forest Regression (RegRf)

A random forest regressor averages many decision trees, each grown on a bootstrap
sample with random feature subsets. Averaging de-correlated trees lowers variance and
usually gives strong, robust predictions with little tuning. `ntree` sets the number
of trees and `mtry` the features considered per split.

**Reference:** Breiman, L. (2001). *Random Forests.* Machine Learning.

```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
```

```python
model = dal.RegRf("target", ntree=100, random_state=1).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["target"], model.predict(test.drop(columns="target")))["metrics"]
```

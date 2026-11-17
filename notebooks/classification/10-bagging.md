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

# Bagging (ClaBagging)

Bagging (bootstrap aggregating) trains several base learners on different bootstrap
samples of the data and combines them by voting. It reduces variance and stabilizes
unstable learners such as decision trees. Random forest is a specialized, feature-
randomized form of bagging; `ClaBagging` is the general version. `nbagg` sets the
number of base estimators.

**Reference:** Breiman, L. (1996). *Bagging Predictors.* Machine Learning.

```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleStratified("Species", seed=42).train_test(iris)
train, test = split["train"], split["test"]
```

```python
model = dal.ClaBagging("Species", slevels, nbagg=25, random_state=1).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```

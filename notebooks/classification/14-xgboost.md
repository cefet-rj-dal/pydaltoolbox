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

# XGBoost (ClaXgboost)

XGBoost is a scalable, regularized gradient boosting machine: it grows an additive
ensemble of trees in which each new tree is fit to the gradient of a differentiable
loss, correcting the errors of the trees already in the model. Second-order (Newton)
boosting, shrinkage and column subsampling make it a strong default for tabular
classification. `nrounds` sets the number of boosting rounds, and the objective
(logistic for two classes, softmax for multiclass) is chosen from the class count.

`xgboost` is an optional dependency: install it with `pip install pydaltoolbox[xgboost]`.

**Reference:** Chen, T., and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting
System.* ACM SIGKDD.

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
model = dal.ClaXgboost("Species", slevels, nrounds=20, random_state=1).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```

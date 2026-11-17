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

# Classification Tuning (ClaTune)

`ClaTune` searches a grid of hyperparameters by k-fold cross-validation and refits the
best configuration on the full training data. It maximizes a classification metric
(default: accuracy). `fit` returns the tuned, fitted learner -- ready to predict.

**Reference:** Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for
Accuracy Estimation and Model Selection.* IJCAI.

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleStratified("Species", seed=42).train_test(iris)
train, test = split["train"], split["test"]

tune = dal.ClaTune(
    dal.ClaKnn("Species", slevels),
    folds=5,
    ranges={"k": [1, 3, 5, 7, 9]},
    seed=42,
)
model = tune.fit(train)
print("chosen k:", model.k)
```

```python
pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], pred)["metrics"]
```

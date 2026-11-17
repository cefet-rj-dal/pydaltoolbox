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

# Naive Bayes (ClaNb)

Naive Bayes applies Bayes' theorem under the "naive" assumption that features are
conditionally independent given the class. Despite this strong assumption it is fast,
needs little data, and is a solid baseline for many problems. `ClaNb` uses the
Gaussian variant, which models each numeric feature with a per-class normal
distribution.

**Reference:** Duda, R. O., Hart, P. E. (1973). *Pattern Classification and Scene
Analysis.* Wiley.

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
model = dal.ClaNb("Species", slevels).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```

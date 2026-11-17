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

# k-Nearest Neighbors (ClaKnn)

k-NN is an instance-based (lazy) learner: it stores the training set and classifies
a new point by majority vote among its `k` closest neighbors. There is no explicit
training phase, which makes it simple and surprisingly effective on well-scaled,
low-dimensional data. Because distances drive the decision, feature scaling (e.g.
`MinMax` or `ZScore`) usually matters.

**Reference:** Cover, T. M., Hart, P. E. (1967). *Nearest Neighbor Pattern
Classification.* IEEE Transactions on Information Theory.

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
model = dal.ClaKnn("Species", slevels, k=5).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```

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

# k-Nearest Neighbors Regression (RegKnn)

k-NN regression predicts the target of a new point as the average of its `k` nearest
neighbors in the training set. It is non-parametric and makes no assumption about the
functional form, but relies on a meaningful distance, so feature scaling matters.
The diabetes features are already scaled, which suits this method.

**Reference:** Cover, T. M., Hart, P. E. (1967). *Nearest Neighbor Pattern
Classification.* IEEE Transactions on Information Theory.

```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
```

```python
model = dal.RegKnn("target", k=5).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["target"], model.predict(test.drop(columns="target")))["metrics"]
```

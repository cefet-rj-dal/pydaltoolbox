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

# Decision Tree Regression (RegDtree)

A regression tree partitions the feature space recursively and predicts, for each
leaf, the mean target of the training points that fall there. It captures non-linear
relationships and interactions and is easy to interpret, but a single deep tree can
overfit -- which motivates the ensembles that build on it.

**Reference:** Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984).
*Classification and Regression Trees.* Wadsworth.

```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
```

```python
model = dal.RegDtree("target", random_state=1).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["target"], model.predict(test.drop(columns="target")))["metrics"]
```

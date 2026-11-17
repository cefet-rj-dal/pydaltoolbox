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

# Majority Baseline (ClaMajority)

The majority classifier always predicts the most frequent class in the training data,
ignoring the features entirely. It is not meant to be competitive -- it is a
**reference floor**. Any real learner should clearly beat it; if it does not, the
features or the setup deserve a second look. On balanced *iris* its accuracy is about
1/3.

**Reference:** Witten, I. H., Frank, E. (2005). *Data Mining: Practical Machine
Learning Tools and Techniques* (the ZeroR baseline).

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
model = dal.ClaMajority("Species", slevels).fit(train)
```

```python
# performance on the held-out test set -- the floor other learners must beat
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```

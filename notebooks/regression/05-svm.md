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

# Support Vector Regression (RegSvm)

Support vector regression fits a function that stays within an `epsilon` tube around
the data while penalizing points outside it, and uses the kernel trick (e.g. RBF) to
model non-linear relationships. It is effective on small-to-medium, well-scaled
datasets. `cost` trades off flatness against tolerated error and `epsilon` sets the
tube width.

**Reference:** Drucker, H., Burges, C., Kaufman, L., Smola, A., Vapnik, V. (1997).
*Support Vector Regression Machines.* NeurIPS.

```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
```

```python
model = dal.RegSvm("target", kernel="rbf", cost=1.0).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["target"], model.predict(test.drop(columns="target")))["metrics"]
```

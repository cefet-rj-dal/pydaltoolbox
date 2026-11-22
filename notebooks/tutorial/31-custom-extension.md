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

# Tutorial 31 - Custom Extension

A researcher may want a method that is not built in, while keeping the same Experiment Line
workflow. This tutorial revisits that idea with a compact custom classifier. The lesson is the
integration pattern, not the specific backend.

## Define the custom classifier (constructor, `fit`, `predict`)
```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import pydaltoolbox as dal


class ClaKnnCustom(dal.Classification):
    def __init__(self, attribute, slevels=None, k=5):
        super().__init__(attribute, slevels)
        self.k = k

    def fit(self, data, **kwargs):
        x = data.drop(columns=self.attribute)
        y = data[self.attribute].astype(str)
        if self.slevels is None:
            self.slevels = sorted(y.unique())
        self.model = KNeighborsClassifier(n_neighbors=self.k).fit(x, y)
        return self

    def predict(self, x, **kwargs):
        proba = self.model.predict_proba(x)
        out = pd.DataFrame(proba, columns=self.model.classes_, index=x.index)
        return out.reindex(columns=self.slevels, fill_value=0.0)
```

## Use the custom learner exactly like a built-in one
```python
iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]

model = ClaKnnCustom("Species", slevels, k=5).fit(train)
pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], pred)["metrics"]
```

Extending the framework should not require rewriting the entire experimental workflow.

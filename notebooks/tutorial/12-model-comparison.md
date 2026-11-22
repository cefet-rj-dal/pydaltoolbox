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

# Tutorial 12 - Model Comparison

Compare multiple learners under the same conditions: same dataset, same split, same metrics,
different models. This is where `pydaltoolbox` works as an experimental framework.

## One shared train/test split
```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleStratified("Species", seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## A small group of learners (baseline, local, tree, ensemble)
```python
models = {
    "majority": dal.ClaMajority("Species", slevels),
    "tree": dal.ClaDtree("Species", slevels, random_state=1),
    "knn": dal.ClaKnn("Species", slevels, k=5),
    "rf": dal.ClaRf("Species", slevels, ntree=10, mtry=2, random_state=1),
}
```

## Fit and evaluate each learner with the same protocol
```python
report = []
for name, model in models.items():
    model.fit(train)
    pred = model.predict(test.drop(columns="Species"))
    m = model.evaluate(test["Species"], pred)["metrics"].copy()
    m.insert(0, "model", name)
    report.append(m)
pd.concat(report, ignore_index=True)
```

The learner changes, but the experiment design stays stable.

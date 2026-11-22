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


## About the method
- `ClaKnn`: k-Nearest Neighbors classifier. Classifies by majority vote among the nearest neighbors in feature space.
- Main parameter: `k`.

## Didactic goal
Keep the same classification line of experiment and change only the learner to a local neighborhood method.

## Environment setup
```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data and inspect.
```python
iris = dal.datasets.iris()
iris.head()
```

## Target `Species` levels and reproducible train/test split.
```python
slevels = list(iris["Species"].cat.categories)
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## Class distribution by split
```python
pd.DataFrame({
    "dataset": iris["Species"].value_counts(),
    "training": train["Species"].value_counts(),
    "test": test["Species"].value_counts(),
}).T
```

## Model configuration and fitting
```python
model = dal.ClaKnn("Species", slevels, k=1).fit(train)
```

## Training evaluation
```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```

## Test evaluation
```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```

## What to observe
- The workflow is the same as in the earlier examples.
- The method-specific change is that prediction depends on local similarity rather than on an explicit parametric or rule-based model.

## References
- Cover, T., and Hart, P. (1967). *Nearest Neighbor Pattern Classification.*

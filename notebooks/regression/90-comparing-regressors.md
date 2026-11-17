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

# Regression - Comparing Models

As with classification, every regressor follows the same Experiment Line contract, so
comparing them is a single loop. This example trains the scikit-learn-backed
regressors on the *diabetes* dataset and ranks them by R2 (higher is better) on the
same held-out test set.

```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
x_test = test.drop(columns="target")
y_test = test["target"]
```

```python
models = {
    "linear": dal.RegLm("target"),
    "decision tree": dal.RegDtree("target", random_state=1),
    "knn": dal.RegKnn("target"),
    "random forest": dal.RegRf("target", random_state=1),
    "svm": dal.RegSvm("target"),
    "mlp": dal.RegMlp("target", size=10, decay=0.01, maxit=2000, random_state=1),
}
```

```python
rows = []
for name, model in models.items():
    model.fit(train)
    m = model.evaluate(y_test, model.predict(x_test))["metrics"].iloc[0]
    rows.append({"model": name, "R2": round(m["R2"], 4), "mse": round(m["mse"], 2)})

results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
results
```

The ranking depends on the dataset, but the point stands: the **same workflow**
fit, predicted and scored every model. Swapping a regressor is a one-line change, and
the uniform `evaluate` returns comparable MSE / sMAPE / R2 for all of them.

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

# Smoothing by Binning (Smoothing family)

Smoothing reduces noise in a numeric attribute by splitting it into `n` bins and replacing
each value with its bin mean — a form of discretization. The strategies differ in how the
bin boundaries are chosen:

- `SmoothingInter` — equal-width intervals (robust boxplot bounds).
- `SmoothingFreq` — equal-frequency bins (quantiles).
- `SmoothingQuantization` — k-means quantization of the values.
- `SmoothingCluster` — class-aware clustering (value + one-hot label).

All follow the transform contract (`fit` → `transform`) and expose `evaluate` to score a
discretization against a class label by entropy (lower = purer bins).

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Discretization and Concept Hierarchy Generation).

```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris["sepal_length"]
x.describe()[["min", "mean", "max"]]
```

Equal-frequency smoothing into 3 bins replaces each value with its bin mean:

```python
sm = dal.SmoothingFreq(n=3).fit(x)
smoothed = sm.transform(x)
pd.Series(smoothed).value_counts().sort_index()   # 3 distinct bin means
```

`fit` can also tune `n` over a range of candidates (elbow of the MSE curve):

```python
tuned = dal.SmoothingFreq(n=range(2, 8)).fit(x)
print("chosen n:", tuned.n)
```

`evaluate` scores the discretization against the class label by entropy — handy to compare
strategies:

```python
rows = []
for name, sm in {
    "inter": dal.SmoothingInter(n=3),
    "freq": dal.SmoothingFreq(n=3),
    "quantization": dal.SmoothingQuantization(n=3, seed=1),
}.items():
    sm.fit(x)
    bins = pd.cut(x, bins=np.unique(sm.interval_adj), include_lowest=True, labels=False)
    rows.append({"strategy": name, "entropy": round(sm.evaluate(bins, iris["Species"])["entropy"], 4)})

pd.DataFrame(rows).sort_values("entropy")
```

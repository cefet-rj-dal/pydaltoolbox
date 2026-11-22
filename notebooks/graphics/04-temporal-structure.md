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

# Temporal & Structure Plots (series, time series, predictions, dendrogram)

These `plot_*` functions cover ordered series and hierarchical structure. They mirror the R
`daltoolbox` ggplot2 graphics on matplotlib / scipy and need the optional `[viz]` extra
(`pip install pydaltoolbox[viz]`). Each returns a `matplotlib.axes.Axes`.

**Reference:** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
Practice* (3rd ed.). OTexts.

```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal

x = np.arange(0, 10, 0.25)
```

## Series

`plot_series` draws one or more series (line + points) over a common x index.

```python
frame = pd.DataFrame({"x": x, "sin": np.sin(x), "cos": np.cos(x)})
ax = dal.plot_series(frame)
ax.figure
```

## Time series

`plot_ts` plots a single series; if `x` is omitted an integer index is used.

```python
ax = dal.plot_ts(x=x, y=np.sin(x), color="#2a78d6")
ax.figure
```

## Time series with predictions

`plot_ts_pred` overlays the fitted (training) values and an optional forecast as dashed lines
over the observed series.

```python
y = np.sin(x) + np.random.default_rng(0).normal(0, 0.1, len(x))
ax = dal.plot_ts_pred(y=y, yadj=np.sin(x[:35]), ypred=np.sin(x[35:]))
ax.figure
```

## Dendrogram

`plot_dendrogram` renders a hierarchical clustering from a SciPy linkage matrix.

```python
from scipy.cluster.hierarchy import linkage

iris = dal.datasets.iris()
z = linkage(iris.drop(columns="Species").to_numpy(), method="ward")
ax = dal.plot_dendrogram(z, labels=False)
ax.figure
```

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

# Time Series Lines (plot_series)

## About the chart
- Time series (lines): points connected by segments; highlights trend and seasonality over an ordered axis.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

## Synthetic series (sine and shifted cosine)
```python
x = np.arange(0, 10, 0.25)
serie = pd.DataFrame({"x": x, "sin": np.sin(x), "cosine": np.cos(x) + 5})
serie.head()
```

## Build a two-line series chart
```python
ax = dal.plot_series(serie)
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Hyndman, R. J., and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

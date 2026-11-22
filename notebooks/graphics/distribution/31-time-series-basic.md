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

# Basic Time Series (plot_ts)

## About the chart
- Simple time series: exploratory visualization of a temporal vector with ordered x-axis and y values.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

## Synthetic series (sine) and basic plot
```python
x = np.arange(0, 10, 0.25)
y = np.sin(x)
ax = dal.plot_ts(x=x, y=y, color="#e34948")
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Hyndman, R. J., and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

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

# Point Chart (plot_points)

## About the chart
- Points: similar to a series, but without connecting lines; good to highlight discrete observations.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

## Synthetic series
```python
x = np.arange(0, 10, 0.25)
serie = pd.DataFrame({"x": x, "sin": np.sin(x), "cosine": np.cos(x) + 5})
serie.head()
```

## Plot points
```python
ax = dal.plot_points(serie)
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.

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

# Density Plot (plot_density)

## About the chart
- Density (kernel density): a smoothed version of the histogram for continuous variables; highlights distribution shapes.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

## Generate variables with distinct distributions
```python
rng = np.random.default_rng(1)
example = pd.DataFrame({
    "exponential": rng.exponential(1.0, 10000),
    "uniform": rng.uniform(2.5, 3.5, 10000),
    "normal": rng.normal(5.0, 1.0, 10000),
})
example.head()
```

## Overlaid density curves
```python
ax = dal.plot_density(example)
ax.figure
```

## References
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis.* Chapman and Hall.
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.

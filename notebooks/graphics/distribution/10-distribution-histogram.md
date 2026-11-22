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

# Histogram (plot_hist)

## About the chart
- Histogram: distributes observations into bins along the x-axis; useful to visualize frequency and skewness.

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

## Histogram of the exponential variable
```python
ax = dal.plot_hist(example[["exponential"]], label_x="exponential", color="#2a78d6")
ax.figure
```

## References
- Freedman, D., and Diaconis, P. (1981). *On the histogram as a density estimator.*
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.

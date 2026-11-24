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

# Exporting a Chart to PDF

## About the feature
- Chart saving: export a figure with `Figure.savefig` to PDF, controlling size and resolution.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

```python
import os
import tempfile
```

## Build a chart
```python
rng = np.random.default_rng(0)
x = np.arange(30)
long = pd.DataFrame({
    "x": np.r_[x, x],
    "value": np.r_[x + rng.normal(0, 1, 30), 0.5 * x + rng.normal(0, 1, 30)],
    "variable": ["a"] * 30 + ["b"] * 30,
})
ax = dal.plot_scatter(long, label_x="x", label_y="value")
ax.figure
```

## Save it as PDF
```python
path = os.path.join(tempfile.gettempdir(), "series.pdf")
ax.figure.savefig(path, bbox_inches="tight")
print("saved", path)
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.

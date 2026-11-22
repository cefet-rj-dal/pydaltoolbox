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

# Radar Chart (plot_radar)

## About the chart
- Radar: displays multiple numeric variables on radial axes from a common origin; useful for comparative profiles.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

```python
iris = dal.datasets.iris()
```

## Mean per numeric variable, shaped as name/value
```python
means = iris.drop(columns="Species").mean()
profile = pd.DataFrame({"name": means.index, "value": means.to_numpy()}).sort_values("name")
profile
```

## Build a radar chart
```python
ax = dal.plot_radar(profile, colors="#2a78d6")
ax.figure
```

## What to observe
- With four variables, the profile is a quadrilateral; in general it is an n-sided polygon, where n is the number of variables.
- `plot_radar` already sets the full drawing area.

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Wilkinson, L. (2005). *The Grammar of Graphics* (2nd ed.). Springer.

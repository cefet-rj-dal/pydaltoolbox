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

# Dendrogram (plot_dendrogram)

## About the chart
- `plot_dendrogram`: dendrogram rendering for hierarchical clustering structures.

## Didactic goal
Bridge clustering and charting: a dendrogram plots merge structure, not points.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

```python
from scipy.cluster.hierarchy import linkage

iris = dal.datasets.iris()
```

## Dendrogram from a SciPy linkage
```python
z = linkage(iris.drop(columns="Species").to_numpy(), method="ward")
ax = dal.plot_dendrogram(z, labels=False, title="Iris dendrogram")
ax.figure
```

## References
- Ward, J. H. (1963). *Hierarchical grouping to optimize an objective function.*

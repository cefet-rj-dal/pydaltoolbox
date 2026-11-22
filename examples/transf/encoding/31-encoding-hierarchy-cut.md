# Hierarchy Cut (HierarchyCut)

## About the transformation
- `HierarchyCut`: converts a numeric attribute into ordered categories defined by cut points.

## Didactic goal
Show how a continuous variable can be turned into a simple hierarchy that is easier to interpret, summarize, or use downstream as a categorical descriptor.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Cut `sepal_length` into ordered levels


```python
iris = dal.datasets.iris()
hc = dal.HierarchyCut(
    "sepal_length",
    breaks=[-float("inf"), 5.5, 6.5, float("inf")],
    labels=["short", "medium", "long"],
)
out = hc.transform(iris)
out["sepal_length_Level"].value_counts()
```




    sepal_length_Level
    medium    61
    short     59
    long      30
    Name: count, dtype: int64



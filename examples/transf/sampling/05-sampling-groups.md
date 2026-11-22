# Group (Cluster) Sampling (SampleGroups)

## About the transformation
- `SampleGroups`: samples whole groups defined by a categorical attribute (cluster sampling).

## Didactic goal
Show a sampling strategy where the unit of selection is the group, not the individual row. Here "cluster" means a pre-existing group in the data, not a group inferred by a clustering algorithm. This matters when preserving within-group structure is more important than preserving the global distribution exactly.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Keep whole `Species` groups together


```python
iris = dal.datasets.iris()
sampled = dal.SampleGroups("Species", n_groups=2, seed=1).transform(iris)
sampled["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    50
    virginica      0
    Name: count, dtype: int64



## What to observe
- Entire species groups are kept or removed together in this toy example.
- This is useful when rows belong to natural blocks that should not be split arbitrarily.

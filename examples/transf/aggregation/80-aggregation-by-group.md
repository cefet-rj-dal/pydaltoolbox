# Aggregation by Group (Aggregation)

## About the transformation
- `Aggregation`: summarizes a dataset by groups through named aggregation expressions.

## Didactic goal
Show that some transformations are not instance-level edits but dataset-level restructuring steps. Aggregation is useful before plotting, reporting, and grouped comparison.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Summarize sepal length by species


```python
iris = dal.datasets.iris()
agg = dal.Aggregation(
    "Species",
    mean_sepal=("sepal_length", "mean"),
    sd_sepal=("sepal_length", "std"),
    n=("sepal_length", "size"),
)
agg.transform(iris)
```




          Species  mean_sepal  sd_sepal   n
    0      setosa       5.006  0.352490  50
    1  versicolor       5.936  0.516171  50
    2   virginica       6.588  0.635880  50



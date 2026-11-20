# Feature Engineering (FeatureGeneration, Aggregation, HierarchyCut)

Three small, native transforms for shaping features:

- `FeatureGeneration` — derive new columns from existing ones.
- `Aggregation` — summarize rows into one row per group.
- `HierarchyCut` — turn a numeric attribute into an ordered categorical level.

**Reference:** Han, J., Kamber, M., Pei, J. (2011). *Data Mining: Concepts and
Techniques* (Data Transformation, Attribute Construction and Aggregation).


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

**FeatureGeneration** — each new column is a callable of the DataFrame or an `eval`
expression:


```python
gen = dal.FeatureGeneration(
    sepal_area=lambda df: df["sepal_length"] * df["sepal_width"],
    petal_ratio="petal_length / petal_width",
)
gen.transform(iris).head()
```




       sepal_length  sepal_width  petal_length  petal_width Species  sepal_area  petal_ratio
    0           5.1          3.5           1.4          0.2  setosa       17.85          7.0
    1           4.9          3.0           1.4          0.2  setosa       14.70          7.0
    2           4.7          3.2           1.3          0.2  setosa       15.04          6.5
    3           4.6          3.1           1.5          0.2  setosa       14.26          7.5
    4           5.0          3.6           1.4          0.2  setosa       18.00          7.0



**Aggregation** — one row per group, with named aggregations (`(column, func)` or a
callable of the group):


```python
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



**HierarchyCut** — discretize a numeric attribute into labeled levels:


```python
hc = dal.HierarchyCut(
    "sepal_length",
    breaks=[-float("inf"), 5.5, 6.5, float("inf")],
    labels=["low", "medium", "high"],
)
hc.transform(iris)["sepal_length_Level"].value_counts()
```




    sepal_length_Level
    medium    61
    low       59
    high      30
    Name: count, dtype: int64



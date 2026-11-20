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

**HierarchyCut** — discretize a numeric attribute into labeled levels:

```python
hc = dal.HierarchyCut(
    "sepal_length",
    breaks=[-float("inf"), 5.5, 6.5, float("inf")],
    labels=["low", "medium", "high"],
)
hc.transform(iris)["sepal_length_Level"].value_counts()
```

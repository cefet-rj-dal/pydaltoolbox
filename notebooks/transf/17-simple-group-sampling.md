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

# Simple and Group Sampling (SampleSimple, SampleGroups)

Two one-shot samplers (they return a sampled subset, unlike `SampleRandom`/`SampleStratified`
which produce train/test splits):

- `SampleSimple` — a simple random sample of rows (or elements), with or without replacement.
- `SampleGroups` — *cluster sampling*: keep all rows of a few randomly chosen groups.

**Reference:** Cochran, W. G. (1977). *Sampling Techniques.* Wiley.

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

**SampleSimple** — draw 10 rows without replacement:

```python
dal.SampleSimple(size=10, seed=1).transform(iris)["Species"].value_counts()
```

With replacement you can draw more rows than the dataset has:

```python
len(dal.SampleSimple(size=300, replace=True, seed=1).transform(iris))
```

**SampleGroups** — keep every row of 2 randomly chosen species (whole groups):

```python
sg = dal.SampleGroups("Species", n_groups=2, seed=1).transform(iris)
sg["Species"].value_counts()
```

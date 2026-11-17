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

# Tutorial 02 - Sampling Strategy

The first tutorial used a single random split. Here we look more closely at *how*
the data is partitioned, because the sampling strategy affects how trustworthy the
evaluation is. `pydaltoolbox` provides sampling objects that follow the same
Experiment Line contract: configure once, then ask for a train/test split or for
k folds.

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris["Species"].value_counts()
```

## Random sampling

`SampleRandom` splits rows uniformly at random. It is the simplest strategy and a
good default when the classes are reasonably balanced. A seed makes the split
reproducible.

```python
sr = dal.SampleRandom(seed=1)
tt = sr.train_test(iris, perc=0.8)
print("train:", len(tt["train"]), " test:", len(tt["test"]))
print(tt["train"]["Species"].value_counts())
```

The same object also produces k folds for cross-validation.

```python
folds = sr.k_fold(iris, k=4)
for i, f in enumerate(folds, start=1):
    print(f"fold {i}: {len(f)} rows")
```

## Stratified sampling

When the class distribution matters -- especially for imbalanced problems --
`SampleStratified` preserves the target proportions in every split. It takes the
name of the target attribute.

```python
ss = dal.SampleStratified("Species", seed=1)
tt = ss.train_test(iris, perc=0.8)
print(tt["train"]["Species"].value_counts(normalize=True).round(3))
```

Each stratified fold keeps all classes represented in the same proportion.

```python
folds = ss.k_fold(iris, k=4)
for i, f in enumerate(folds, start=1):
    counts = f["Species"].value_counts().sort_index().to_dict()
    print(f"fold {i}: {counts}")
```

## Building a split from folds

Given a list of folds, `train_test_from_folds` uses the k-th fold (1-based) as the
test set and concatenates the rest as the training set. This is the building block
of a manual cross-validation loop.

```python
split = dal.train_test_from_folds(folds, k=1)
print("train:", len(split["train"]), " test:", len(split["test"]))
```

The sampling object is just another Experiment Line component: the rest of the
workflow (fit, predict, evaluate) stays exactly the same regardless of which
sampler you choose.

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

# Feature Selection by Information Gain (FeatureSelectionInfoGain)

Information gain measures how much knowing a feature reduces the entropy (uncertainty)
of the target. It ranks features for a classification target; numeric predictors are
discretized into quantile bins first. Keep the top features (`top`) or those above a
minimum gain (`cutoff`).

**Reference:** Quinlan, J. R. (1986). *Induction of Decision Trees.* Machine Learning.

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
fs = dal.FeatureSelectionInfoGain("Species", top=2).fit(iris)
fs.ranking
```

The petal measurements are the most informative about the species.

```python
print("selected:", fs.selected)
fs.transform(iris).head()
```

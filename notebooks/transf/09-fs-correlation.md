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

# Feature Selection by Correlation (FeatureSelectionCorr)

Highly correlated predictors carry redundant information and can destabilize some
models. This filter removes, from each pair whose absolute correlation exceeds
`cutoff`, the feature that is on average more correlated with the others -- keeping a
less-redundant subset. It is unsupervised (no target needed).

**Reference:** Kuhn, M., Johnson, K. (2013). *Applied Predictive Modeling* (the
`findCorrelation` heuristic).

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
fs = dal.FeatureSelectionCorr(cutoff=0.9).fit(iris)
print("removed:", fs.remove)
print("selected:", fs.selected)
```

```python
fs.transform(iris).head()
```

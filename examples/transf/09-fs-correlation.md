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

    removed: ['petal_length']
    selected: ['sepal_length', 'sepal_width', 'petal_width']
    


```python
fs.transform(iris).head()
```




       sepal_length  sepal_width  petal_width Species
    0           5.1          3.5          0.2  setosa
    1           4.9          3.0          0.2  setosa
    2           4.7          3.2          0.2  setosa
    3           4.6          3.1          0.2  setosa
    4           5.0          3.6          0.2  setosa



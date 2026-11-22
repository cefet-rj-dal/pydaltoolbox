# Feature Selection by RELIEF (FeatureSelectionRelief)

ReliefF is an instance-based filter. For each of `m` randomly sampled instances it finds
the nearest neighbour of the same class (nearest hit) and of a different class (nearest
miss) in the normalized feature space, and updates each feature's weight by
`-|x - nearHit| + |x - nearMiss|`. Features that keep same-class instances close and
different-class instances apart score high, so Relief captures relevance including local
feature interactions. The target is treated as a class factor. This is a native numpy
port of the R `feature_selection_relief` (no `skrebate` dependency).

**References:** Kira, K. & Rendell, L. A. (1992); Kononenko, I. (1994).


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
iris = iris.assign(label=(iris["Species"] == "versicolor").astype(int))
iris = iris.drop(columns=["Species"])

fs = dal.FeatureSelectionRelief("label", m=50, seed=42).fit(iris)
fs.ranking
```




            feature     score
    0   petal_width  0.197583
    1  petal_length  0.160610
    2   sepal_width  0.066250
    3  sepal_length  0.036833




```python
fs = dal.FeatureSelectionRelief("label", top=2, seed=42).fit(iris)
print("selected:", fs.selected)
fs.transform(iris).head()
```

    selected: ['petal_width', 'petal_length']
    




       petal_length  petal_width  label
    0           1.4          0.2      0
    1           1.4          0.2      0
    2           1.3          0.2      0
    3           1.5          0.2      0
    4           1.4          0.2      0



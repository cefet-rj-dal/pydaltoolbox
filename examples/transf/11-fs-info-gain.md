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




            feature     score
    0   petal_width  1.369369
    1  petal_length  1.324664
    2  sepal_length  0.646951
    3   sepal_width  0.357132



The petal measurements are the most informative about the species.


```python
print("selected:", fs.selected)
fs.transform(iris).head()
```

    selected: ['petal_width', 'petal_length']
    




       petal_length  petal_width Species
    0           1.4          0.2  setosa
    1           1.4          0.2  setosa
    2           1.3          0.2  setosa
    3           1.5          0.2  setosa
    4           1.4          0.2  setosa



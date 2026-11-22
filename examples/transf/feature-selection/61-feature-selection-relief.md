# Feature Selection with Relief (FeatureSelectionRelief)

## About the transformation
Relief estimates feature relevance by comparing each observation with its nearest hit and nearest miss. Useful for ranking predictors in classification problems.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Rank features and keep the top 2


```python
iris = dal.datasets.iris()
fs = dal.FeatureSelectionRelief("Species", top=2, m=50, seed=1).fit(iris)
fs.transform(iris).head()
```




       petal_length  petal_width Species
    0           1.4          0.2  setosa
    1           1.4          0.2  setosa
    2           1.3          0.2  setosa
    3           1.5          0.2  setosa
    4           1.4          0.2  setosa



## References
- Kononenko, I. (1994). *Estimating attributes: Analysis and extensions of Relief.* ECML.

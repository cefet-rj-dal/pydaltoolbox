# Feature Selection with Information Gain (FeatureSelectionInfoGain)

## About the transformation
Information Gain measures the reduction in target entropy provided by each feature. This example ranks the predictors and keeps the most informative ones for a categorical target.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Rank features and keep the top 2


```python
iris = dal.datasets.iris()
fs = dal.FeatureSelectionInfoGain("Species", top=2).fit(iris)
fs.transform(iris).head()
```




       petal_length  petal_width Species
    0           1.4          0.2  setosa
    1           1.4          0.2  setosa
    2           1.3          0.2  setosa
    3           1.5          0.2  setosa
    4           1.4          0.2  setosa



## References
- Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning.* Morgan Kaufmann.

# Predictive Imputation (ImputationTree)

Instead of filling missing values with a single statistic (as `ImputationSimple` does),
*predictive* imputation learns the missing column from the others. `ImputationTree` fits a
decision tree on the rows where the target is observed, using the source columns as
predictors (source gaps are first completed with a simple imputation so the tree can always
be trained and applied). A regressor is used for a numeric target, a classifier otherwise.

`ImputationPredictive` is the extensible base; `ImputationTree` is the concrete tree-based
implementation.

**Reference:** van Buuren, S., Groothuis-Oudshoorn, K. (2011). *mice: Multivariate
Imputation by Chained Equations in R.* / Breiman, L. et al. (1984). *CART.*


```python
import numpy as np

import pydaltoolbox as dal

iris = dal.datasets.iris().copy()
iris.loc[[2, 10, 25], "sepal_length"] = np.nan
print("missing in sepal_length:", int(iris["sepal_length"].isna().sum()))
```

    missing in sepal_length: 3
    

Fit on the observed rows and impute the missing ones from the other columns:


```python
imp = dal.ImputationTree("sepal_length", random_state=1).fit(iris)
out = imp.transform(iris)
print("missing after imputation:", int(out["sepal_length"].isna().sum()))
out.loc[[2, 10, 25], "sepal_length"]
```

    missing after imputation: 0
    




    2     4.520000
    10    5.233333
    25    4.862500
    Name: sepal_length, dtype: float64



A categorical target works too (the tree becomes a classifier):


```python
iris2 = dal.datasets.iris().copy()
iris2["Species"] = iris2["Species"].astype(object)
iris2.loc[[3, 15, 40], "Species"] = np.nan
dal.ImputationTree("Species", random_state=1).fit(iris2).transform(iris2).loc[[3, 15, 40], "Species"]
```




    3     setosa
    15    setosa
    40    setosa
    Name: Species, dtype: object



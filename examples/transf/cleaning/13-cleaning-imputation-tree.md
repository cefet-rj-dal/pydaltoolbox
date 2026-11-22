# Tree-based Imputation (ImputationTree)

## About the transformation
- `ImputationTree`: tree-based predictive imputation for one target column at a time, using the other attributes as predictors.

## Didactic goal
Contrast model-based imputation with simpler fill rules such as mean or median replacement.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Create missing values in one target column, then impute


```python
iris = dal.datasets.iris().copy()
iris.loc[[1, 9, 24], "sepal_length"] = None
print("missing before:", int(iris["sepal_length"].isna().sum()))
imp = dal.ImputationTree("sepal_length", random_state=1).fit(iris)
out = imp.transform(iris)
print("missing after:", int(out["sepal_length"].isna().sum()))
out["sepal_length"].describe()
```

    missing before: 3
    missing after: 0
    




    count    150.000000
    mean       5.843174
    std        0.828781
    min        4.300000
    25%        5.100000
    50%        5.800000
    75%        6.400000
    max        7.900000
    Name: sepal_length, dtype: float64



## References
- Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). *Classification and Regression Trees.*
- van Buuren, S., and Groothuis-Oudshoorn, K. (2011). *mice: Multivariate Imputation by Chained Equations in R.*

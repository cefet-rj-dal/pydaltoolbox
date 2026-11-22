# NA Removal (NaRemoval)

## About the transformation
- `NaRemoval`: drops rows with missing values. Useful for initial cleanup when imputation is not desired; use with care as it may bias results if missingness is not completely at random.

## Didactic goal
Focus on what changes in the dataset after each operation.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Insert an artificial NA and remove rows with NA


```python
iris = dal.datasets.iris().copy()
iris.loc[1, "sepal_length"] = None
print("rows with NA:", len(iris))
clean = dal.NaRemoval().transform(iris)
print("rows after removal:", len(clean))
```

    rows with NA: 150
    rows after removal: 149
    

## References
- Little, R. J. A., and Rubin, D. B. (2002). *Statistical Analysis with Missing Data* (2nd ed.). Wiley.

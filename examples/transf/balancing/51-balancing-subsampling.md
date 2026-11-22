# Class Undersampling (BalSubsampling)

## About the transformation
- `BalSubsampling`: random undersampling that reduces all classes to the minority count by sampling without replacement.

## Didactic goal
Show the fastest way to reduce class imbalance and highlight its tradeoff: undersampling is simple and often effective, but it discards information from the majority class.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Create an imbalanced dataset


```python
iris = dal.datasets.iris()
imbalanced = iris.iloc[list(range(0, 50)) + list(range(50, 71)) + list(range(100, 111))]
imbalanced["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    21
    virginica     11
    Name: count, dtype: int64



## Apply undersampling and inspect the new class distribution


```python
dal.BalSubsampling("Species", seed=1).transform(imbalanced)["Species"].value_counts()
```




    Species
    setosa        11
    versicolor    11
    virginica     11
    Name: count, dtype: int64



## What to observe
- The balanced result is obtained by removing majority-class rows, not by creating new minority rows.
- As with any balancing transformation, apply it to training data only.

## References
- He, H., and Garcia, E. A. (2009). *Learning from Imbalanced Data.* IEEE TKDE.

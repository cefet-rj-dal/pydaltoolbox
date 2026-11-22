# Class Balancing via Sampling (SampleBalance)

## About the transformation
- `SampleBalance`: class balancing by up-sampling or down-sampling through the sampling interface.

## Didactic goal
Compare this general balancing helper with the more explicit oversampling and subsampling examples. The emphasis is on class-count correction, not on synthetic data generation.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Create an imbalanced subset and then rebalance it


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



## Down-sampling reduces every class to the minority count


```python
dal.SampleBalance("Species", method="down", seed=1).transform(imbalanced)["Species"].value_counts()
```




    Species
    setosa        11
    versicolor    11
    virginica     11
    Name: count, dtype: int64



## Up-sampling replicates minority classes up to the majority count


```python
dal.SampleBalance("Species", method="up", seed=1).transform(imbalanced)["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    50
    virginica     50
    Name: count, dtype: int64



# Random Sampling (SampleRandom)

## About sampling
- `SampleRandom`: splits train/test sets and creates folds via random draws, preserving only expected proportions on average.

## Didactic goal
Interpret sampling as an experimental design choice, not as a routine preamble. A weak split can compromise even a good learner.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load dataset and view class distribution


```python
iris = dal.datasets.iris()
iris["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    50
    virginica     50
    Name: count, dtype: int64



## Split into train/test with random sampling (proportions may vary by draw)


```python
tt = dal.SampleRandom(seed=1).train_test(iris)
pd.DataFrame({
    "training": tt["train"]["Species"].value_counts(),
    "test": tt["test"]["Species"].value_counts(),
})
```




                training  test
    Species                   
    setosa            39    11
    versicolor        40    10
    virginica         41     9



## Create random k-folds and check distribution per fold


```python
folds = dal.SampleRandom(seed=1).k_fold(iris, 4)
pd.DataFrame([f["Species"].value_counts() for f in folds])
```




    Species  setosa  versicolor  virginica
    count        14          12         12
    count        16          10         12
    count         8          14         15
    count        12          14         11



## What to observe
- Training and test class counts are close to the original distribution, but not guaranteed to match it exactly.
- This variability is the main reason random sampling can behave differently across repeated runs.

## References
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI.

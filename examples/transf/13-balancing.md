# Class Balancing (SampleBalance, BalSubsampling, BalOversampling)

When classes are imbalanced, many learners become biased toward the majority. These
transforms rebalance the training data:

- `SampleBalance` - down-sample to the minority count, or up-sample to the majority.
- `BalSubsampling` - undersample every class to the minority count.
- `BalOversampling` - oversample to the majority count, by replication (`random`) or by
  generating synthetic examples (`smote`).

Only the training set should be balanced; the test set must keep its real distribution.

**Reference:** Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002).
*SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
imbalanced = pd.concat([
    iris[iris["Species"] == "setosa"].head(50),
    iris[iris["Species"] == "versicolor"].head(20),
    iris[iris["Species"] == "virginica"].head(10),
]).reset_index(drop=True)
imbalanced["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    20
    virginica     10
    Name: count, dtype: int64



Down-sampling reduces every class to the minority count (10):


```python
dal.SampleBalance("Species", method="down", seed=1).transform(imbalanced)["Species"].value_counts()
```




    Species
    setosa        10
    versicolor    10
    virginica     10
    Name: count, dtype: int64



SMOTE oversampling grows every class to the majority count (50) using synthetic
minority examples:


```python
dal.BalOversampling("Species", method="smote", k=5, seed=1).transform(imbalanced)["Species"].value_counts()
```




    Species
    setosa        50
    versicolor    50
    virginica     50
    Name: count, dtype: int64



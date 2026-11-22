# Baseline — Majority (ClaMajority)

## About the method
- `ClaMajority`: baseline classifier that always predicts the most frequent class observed during training. Useful as a minimum performance reference.

## Didactic goal
Establish the standard classification line of experiment used throughout this family of examples. Later files should be read as variations of this same workflow.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data and inspect.


```python
iris = dal.datasets.iris()
iris.head()
```




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



## Target `Species` levels and reproducible train/test split.


```python
slevels = list(iris["Species"].cat.categories)
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## Class distribution by split


```python
pd.DataFrame({
    "dataset": iris["Species"].value_counts(),
    "training": train["Species"].value_counts(),
    "test": test["Species"].value_counts(),
}).T
```




    Species   setosa  versicolor  virginica
    dataset       50          50         50
    training      39          40         41
    test          11          10          9



## Model configuration and fitting


```python
model = dal.ClaMajority("Species", slevels).fit(train)
```

## Training evaluation


```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity  f1
    0  0.341667   0  81   0  39        NaN     0.0          0.0          1.0 NaN



## Test evaluation


```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity  f1
    0       0.3   0  19   0  11        NaN     0.0          0.0          1.0 NaN



## What to observe
- This learner defines the minimum bar that stronger classifiers should beat.
- The line of experiment already contains everything needed for the later classification examples.

## References
- Witten, I. H., Frank, E., Hall, M. A., and Pal, C. J. (2016). *Data Mining: Practical Machine Learning Tools and Techniques* (4th ed.).

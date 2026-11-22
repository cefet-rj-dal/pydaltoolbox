# Decision Tree (ClaDtree)

## About the method
- `ClaDtree`: decision tree for classification. Recursively splits explanatory variables to separate classes, yielding an interpretable model.

## Didactic goal
Keep exactly the same classification line of experiment introduced by the baseline and change only the learner to an interpretable supervised model.

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
model = dal.ClaDtree("Species", slevels, random_state=1).fit(train)
```

## Training evaluation


```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  39  81   0   0        1.0     1.0          1.0          1.0  1.0



## Test evaluation


```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.933333  11  19   0   0        1.0     1.0          1.0          1.0  1.0



## What to observe
- The experiment structure is unchanged from the baseline example.
- The difference now is that the learner can express feature-based decision rules instead of only the majority class.

## References
- Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). *Classification and Regression Trees.*

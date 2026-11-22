# Tutorial 01 - First Experiment

Understand the basic rhythm of a `pydaltoolbox` experiment. In most supervised tasks the same
cycle appears again and again: prepare the data, split it, configure a learner, fit the model,
predict on unseen cases, and evaluate the result.

## Load data and identify the target


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
slevels
```




    ['setosa', 'versicolor', 'virginica']



## Train/test split


```python
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## Configure and fit a first learner (a decision tree)


```python
model = dal.ClaDtree("Species", slevels, random_state=1).fit(train)
```

## Evaluate on training and test data


```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  39  81   0   0        1.0     1.0          1.0          1.0  1.0




```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.933333  11  19   0   0        1.0     1.0          1.0          1.0  1.0



This structure is the foundation of the package: later tutorials change the sampling strategy,
the preprocessing, or the learner, but the overall logic stays stable.

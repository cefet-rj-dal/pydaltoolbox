# Tutorial 10 - Baseline Classification

Before testing advanced learners, establish a simple reference that answers a basic question:
is the stronger model actually adding value?

## Prepare the problem once so both learners face the same data


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleStratified("Species", seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## The majority baseline (intentionally weak, very informative)


```python
baseline = dal.ClaMajority("Species", slevels).fit(train)
base_pred = baseline.predict(test.drop(columns="Species"))
baseline.evaluate(test["Species"], base_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.333333  10   0  20   0   0.333333     1.0          1.0          0.0  0.5



## A decision tree on the same split


```python
tree = dal.ClaDtree("Species", slevels, random_state=1).fit(train)
tree_pred = tree.predict(test.drop(columns="Species"))
tree.evaluate(test["Species"], tree_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



Always compare a more elaborate model with a transparent baseline, not only with other
elaborate models.

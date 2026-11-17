# Tutorial 01 - First Experiment

The first objective is simple: understand the basic rhythm of a `pydaltoolbox`
experiment. In most supervised tasks the same cycle appears again and again:
prepare the data, split it, configure a learner, fit the model, predict on unseen
cases, and evaluate the result.

This tutorial uses the small and familiar *iris* dataset so the focus stays on the
workflow, not on the domain.


```python
import pydaltoolbox as dal

print("pydaltoolbox", dal.__version__)
```

    pydaltoolbox 0.1.0
    

Start by loading a dataset and identifying the target variable. In classification
it is important to know the possible class labels before training a learner.


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




```python
slevels = list(iris["Species"].cat.categories)
slevels
```




    ['setosa', 'versicolor', 'virginica']



Now create a training set and a test set. The training data is used to fit the
learner; the test data estimates how the model behaves on new cases.


```python
sr = dal.SampleRandom(seed=42)
split = sr.train_test(iris)
train, test = split["train"], split["test"]
print("train:", len(train), " test:", len(test))
```

    train: 120  test: 30
    

With the split ready, define a first learner. A decision tree is a good starting
point because it is fast and easy to interpret.


```python
model = dal.ClaDtree("Species", slevels, random_state=42)
model = model.fit(train)
```

After training, check performance on both the training data and the test data. The
comparison helps reveal whether the learner is memorizing the training set or
generalizing. `predict()` returns class scores (one column per class), so
`evaluate()` can be called directly with the true labels.

Note the clean habit: pass only the feature columns to `predict()`.


```python
train_prediction = model.predict(train.drop(columns="Species"))
train_eval = model.evaluate(train["Species"], train_prediction)
train_eval["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.983333  37  83   0   0        1.0     1.0          1.0          1.0  1.0




```python
test_prediction = model.predict(test.drop(columns="Species"))
test_eval = model.evaluate(test["Species"], test_prediction)
test_eval["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  13  17   0   0        1.0     1.0          1.0          1.0  1.0



This structure is the foundation of the package. Later tutorials mainly change the
sampling strategy, the preprocessing steps, or the learner itself, but the overall
logic -- fit, action/predict, evaluate -- remains stable.

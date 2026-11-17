# Decision Tree (ClaDtree)

A decision tree splits the feature space recursively, choosing at each node the test
that best separates the classes (by impurity reduction). The result is a small set
of human-readable rules, which makes trees a strong first choice when
**interpretability** matters. They train fast and are the building block of
ensembles such as random forests and boosting.

**Reference:** Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984).
*Classification and Regression Trees.* Wadsworth.


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleStratified("Species", seed=42).train_test(iris)
train, test = split["train"], split["test"]
```


```python
model = dal.ClaDtree("Species", slevels, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.933333  10  20   0   0        1.0     1.0          1.0          1.0  1.0



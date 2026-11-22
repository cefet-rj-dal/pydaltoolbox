# Custom Classification

The goal is to show how a classifier can be customized with very little integration code:
define a constructor, implement `fit` to train, and implement `predict` to return a
class-score table (one column per class). The downstream sampling and evaluation workflow is
then identical to a built-in learner. Here the custom classifier wraps scikit-learn's
k-nearest-neighbors backend.

## A custom classifier subclassing `Classification`


```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import pydaltoolbox as dal


class ClaKnnCustom(dal.Classification):
    def __init__(self, attribute, slevels=None, k=5):
        super().__init__(attribute, slevels)
        self.k = k

    def fit(self, data, **kwargs):
        x = data.drop(columns=self.attribute)
        y = data[self.attribute].astype(str)
        if self.slevels is None:
            self.slevels = sorted(y.unique())
        self.model = KNeighborsClassifier(n_neighbors=self.k).fit(x, y)
        return self

    def predict(self, x, **kwargs):
        proba = self.model.predict_proba(x)
        out = pd.DataFrame(proba, columns=self.model.classes_, index=x.index)
        return out.reindex(columns=self.slevels, fill_value=0.0)
```

## Same Experiment Line workflow as a built-in learner


```python
iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleRandom(seed=1).train_test(iris)
train, test = split["train"], split["test"]

model = ClaKnnCustom("Species", slevels, k=5).fit(train)
```

## Training and test evaluation


```python
train_pred = model.predict(train.drop(columns="Species"))
model.evaluate(train["Species"], train_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0     0.975  39  81   0   0        1.0     1.0          1.0          1.0  1.0




```python
test_pred = model.predict(test.drop(columns="Species"))
model.evaluate(test["Species"], test_pred)["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  11  19   0   0        1.0     1.0          1.0          1.0  1.0



## What to observe
- The integration contract is explicit: constructor, `fit`, and `predict`.
- For classification, `predict` returns one column per class (scores), not only hard labels.
- After the contract is implemented, the workflow matches any built-in learner example.

## References
- Cover, T., and Hart, P. (1967). *Nearest Neighbor Pattern Classification.*

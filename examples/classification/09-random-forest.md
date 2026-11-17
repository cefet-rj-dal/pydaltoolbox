# Random Forest (ClaRf)

A random forest is an ensemble of decision trees, each grown on a bootstrap sample
and a random subset of features. Averaging many de-correlated trees reduces variance
and usually yields strong, robust accuracy with little tuning. `ntree` sets the
number of trees and `mtry` the number of features considered per split.

**Reference:** Breiman, L. (2001). *Random Forests.* Machine Learning.


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
model = dal.ClaRf("Species", slevels, ntree=100, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



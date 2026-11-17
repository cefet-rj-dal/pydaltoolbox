# CART Decision Tree (ClaRpart)

`ClaRpart` is the CART-style recursive-partitioning tree (the analog of R's `rpart`).
It uses the Gini impurity to grow binary splits. In `pydaltoolbox` both `ClaDtree`
and `ClaRpart` are backed by scikit-learn's CART implementation; the class is kept
separate to preserve the familiar daltoolbox API and to let the two be configured
and tuned independently.

**Reference:** Therneau, T., Atkinson, E. (1997). *An Introduction to Recursive
Partitioning Using the RPART Routines.* (CART: Breiman et al., 1984.)


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
model = dal.ClaRpart("Species", slevels, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



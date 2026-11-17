# Support Vector Machine (ClaSvm)

An SVM finds the decision boundary that maximizes the margin between classes. With
the kernel trick (e.g. the RBF kernel) it models non-linear boundaries while keeping
a convex optimization problem. SVMs are strong on small-to-medium datasets. `cost`
controls the margin/error trade-off and `gamma` the kernel reach. Features are
standardized inside the model (as R's `e1071::svm` does by default), so no separate
scaling step is needed.

**Reference:** Cortes, C., Vapnik, V. (1995). *Support-Vector Networks.* Machine
Learning.


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
model = dal.ClaSvm("Species", slevels, kernel="rbf", random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



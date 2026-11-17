# Boosting (ClaBoosting)

Boosting builds an ensemble sequentially: each new weak learner focuses on the
examples the previous ones got wrong, and the members are combined by weighted vote.
It reduces bias and often reaches high accuracy. `ClaBoosting` uses AdaBoost (SAMME
for multi-class); `mfinal` sets the number of boosting rounds.

**Reference:** Freund, Y., Schapire, R. (1997). *A Decision-Theoretic Generalization
of On-Line Learning and an Application to Boosting.* / Hastie et al. (2009),
*Multi-class AdaBoost.*


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
model = dal.ClaBoosting("Species", slevels, mfinal=50, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.933333  10  20   0   0        1.0     1.0          1.0          1.0  1.0



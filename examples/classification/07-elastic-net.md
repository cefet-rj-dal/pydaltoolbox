# Elastic-Net Logistic Regression (ClaGlmnet)

`ClaGlmnet` is a regularized logistic regression that blends L1 (lasso) and L2
(ridge) penalties -- the elastic net. Regularization shrinks coefficients, which
helps when features are many or correlated, and the L1 part can drive some
coefficients to zero (implicit feature selection). The `alpha` parameter sets the
L1/L2 mix (0 = ridge, 1 = lasso).

**Reference:** Zou, H., Hastie, T. (2005). *Regularization and Variable Selection
via the Elastic Net.* Journal of the Royal Statistical Society, Series B.


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
model = dal.ClaGlmnet("Species", slevels, alpha=0.5, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



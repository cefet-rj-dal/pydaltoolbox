# Logistic Regression / GLM (ClaGlm)

Logistic regression is a generalized linear model for classification: it models the
log-odds of the classes as a linear function of the features. It is interpretable
(coefficients indicate direction and strength of each feature) and a dependable
linear baseline. `ClaGlm` wraps scikit-learn's `LogisticRegression`.

**Reference:** Nelder, J. A., Wedderburn, R. W. M. (1972). *Generalized Linear
Models.* Journal of the Royal Statistical Society, Series A.


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
model = dal.ClaGlm("Species", slevels, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Multinomial Logistic Regression (ClaMultinom)

The multinomial logit generalizes logistic regression to more than two classes,
estimating the probability of each class directly through a softmax over linear
scores. It is a natural interpretable baseline for multi-class problems such as
*iris*. `ClaMultinom` wraps scikit-learn's `LogisticRegression` in its multinomial
mode.

**Reference:** McFadden, D. (1973). *Conditional Logit Analysis of Qualitative
Choice Behavior.* In Frontiers in Econometrics.

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
model = dal.ClaMultinom("Species", slevels, random_state=1).fit(train)
```

```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```

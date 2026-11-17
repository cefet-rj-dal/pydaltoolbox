# Multi-Layer Perceptron (ClaMlp)

A multi-layer perceptron is a feed-forward neural network trained by backpropagation.
A single hidden layer can already model non-linear boundaries, at the cost of more
tuning (layer size, regularization, iterations) and sensitivity to feature scaling.
`size` sets the hidden-layer width and `decay` the L2 regularization.

**Reference:** Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). *Learning
Representations by Back-Propagating Errors.* Nature.


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
model = dal.ClaMlp("Species", slevels, size=5, decay=0.0, maxit=1000, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["Species"], model.predict(test.drop(columns="Species")))["metrics"]
```




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



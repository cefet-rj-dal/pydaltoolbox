# Multi-Layer Perceptron Regression (RegMlp)

An MLP regressor is a feed-forward neural network trained by backpropagation to
predict a numeric target. A single hidden layer can approximate non-linear functions,
at the cost of tuning (layer size, regularization, iterations) and sensitivity to
scaling. `size` sets the hidden-layer width and `decay` the L2 regularization.

**Reference:** Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). *Learning
Representations by Back-Propagating Errors.* Nature.


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
```


```python
model = dal.RegMlp("target", size=10, decay=0.01, maxit=2000, random_state=1).fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["target"], model.predict(test.drop(columns="target")))["metrics"]
```




               mse     smape        R2
    0  6861.638563  0.399946 -0.077139



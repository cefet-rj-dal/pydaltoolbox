# Support Vector Regression (RegSvm)

Support vector regression fits a function that stays within an `epsilon` tube around
the data while penalizing points outside it, and uses the kernel trick (e.g. RBF) to
model non-linear relationships. It is effective on small-to-medium datasets. `cost`
trades off flatness against tolerated error and `epsilon` sets the tube width.
Features and target are standardized inside the model (as R's `e1071::svm` does by
default), which is essential for SVR on targets with large magnitude.

**Reference:** Drucker, H., Burges, C., Kaufman, L., Smola, A., Vapnik, V. (1997).
*Support Vector Regression Machines.* NeurIPS.


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

data = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(data)
train, test = split["train"], split["test"]
```


```python
model = dal.RegSvm("target", kernel="rbf").fit(train)
```


```python
# performance on the held-out test set
model.evaluate(test["target"], model.predict(test.drop(columns="target")))["metrics"]
```




               mse     smape        R2
    0  4339.272252  0.349317  0.318822



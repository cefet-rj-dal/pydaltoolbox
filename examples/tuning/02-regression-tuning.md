# Regression Tuning (RegTune)

`RegTune` tunes a regressor by k-fold cross-validation, selecting the configuration
with the lowest mean squared error, and refits it on the full training data. `fit`
returns the tuned, fitted regressor.

**Reference:** Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for
Accuracy Estimation and Model Selection.* IJCAI.


```python
import pydaltoolbox as dal

diabetes = dal.datasets.diabetes()
split = dal.SampleRandom(seed=42).train_test(diabetes)
train, test = split["train"], split["test"]

tune = dal.RegTune(dal.RegKnn("target"), folds=5, ranges={"k": [3, 5, 10, 20, 30]}, seed=42)
model = tune.fit(train)
print("chosen k:", model.k)
```

    chosen k: 10
    


```python
pred = model.predict(test.drop(columns="target"))
model.evaluate(test["target"], pred)["metrics"]
```




               mse     smape        R2
    0  3836.190225  0.337667  0.397795



# Tutorial 20 - End-to-End Classification Pipeline

Put the pieces together: split the data, normalize numeric attributes (fitting the normalizer
on training data only), fit a learner, and evaluate.

## Split first, so preprocessing is fit on training data only


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)
split = dal.SampleStratified("Species", seed=1).train_test(iris)
train, test = split["train"], split["test"]
```

## Fit normalization on train; apply to both sets (no leakage)


```python
norm = dal.MinMax().fit(train)
train_norm = norm.transform(train)
test_norm = norm.transform(test)
```

## Fit a scale-sensitive classifier and evaluate


```python
model = dal.ClaKnn("Species", slevels, k=5).fit(train_norm)

train_pred = model.predict(train_norm.drop(columns="Species"))
print(model.evaluate(train_norm["Species"], train_pred)["metrics"])

test_pred = model.predict(test_norm.drop(columns="Species"))
model.evaluate(test_norm["Species"], test_pred)["metrics"]
```

       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0     0.975  40  80   0   0        1.0     1.0          1.0          1.0  1.0
    




       accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  0.966667  10  20   0   0        1.0     1.0          1.0          1.0  1.0



## What to observe
- The split happens before preprocessing because preprocessing objects should learn only from training data.
- The same fitted normalization object is reused on both sets.
- This sequence is a compact template for many real classification studies.

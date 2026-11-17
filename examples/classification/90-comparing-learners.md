# Classification - Comparing Learners

The central idea of the Experiment Line is a **uniform contract**: every classifier
is configured the same way and exposes the same `fit`, `predict` and `evaluate`
methods. Swapping one learner for another is a one-line change, which makes fair
comparisons easy.

This example trains several scikit-learn-backed classifiers on *iris* and compares
their accuracy on the same stratified test set.


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")  # keep the didactic output clean
```

Prepare a single stratified split so every learner is judged on the same data.


```python
iris = dal.datasets.iris()
slevels = list(iris["Species"].cat.categories)

split = dal.SampleStratified("Species", seed=42).train_test(iris)
train, test = split["train"], split["test"]
x_test = test.drop(columns="Species")
y_test = test["Species"]
print("train:", len(train), " test:", len(test))
```

    train: 120  test: 30
    

Configure a family of learners. Note how uniform the construction is -- the only
thing that changes is the class.


```python
learners = {
    "majority (baseline)": dal.ClaMajority("Species", slevels),
    "naive bayes": dal.ClaNb("Species", slevels),
    "knn": dal.ClaKnn("Species", slevels),
    "decision tree": dal.ClaDtree("Species", slevels, random_state=1),
    "logistic (glm)": dal.ClaGlm("Species", slevels, random_state=1),
    "svm": dal.ClaSvm("Species", slevels, random_state=1),
    "random forest": dal.ClaRf("Species", slevels, random_state=1),
    "bagging": dal.ClaBagging("Species", slevels, random_state=1),
    "boosting": dal.ClaBoosting("Species", slevels, random_state=1),
    "mlp": dal.ClaMlp("Species", slevels, random_state=1),
}
```

Because the contract is identical, a single loop fits, predicts and evaluates every
learner.


```python
rows = []
for name, model in learners.items():
    model.fit(train)
    evaluation = model.evaluate(y_test, model.predict(x_test))
    rows.append({"model": name, "accuracy": round(evaluation["metrics"].iloc[0]["accuracy"], 4)})

results = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
results
```




                     model  accuracy
    0                  knn    1.0000
    1          naive bayes    0.9667
    2              bagging    0.9667
    3       logistic (glm)    0.9667
    4                  svm    0.9667
    5        random forest    0.9667
    6             boosting    0.9333
    7        decision tree    0.9333
    8                  mlp    0.8667
    9  majority (baseline)    0.3333



The majority baseline sits at the bottom (it ignores the features), while the real
learners cluster near the top. The important point is not which model wins on this
small dataset, but that the **same workflow** evaluated all of them -- the Experiment
Line made the comparison trivial and fair.

Other learners follow exactly the same pattern: `ClaRpart`, `ClaMultinom` and
`ClaGlmnet`.

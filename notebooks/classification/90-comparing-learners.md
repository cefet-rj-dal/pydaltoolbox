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

The majority baseline sits at the bottom (it ignores the features), while the real
learners cluster near the top. The important point is not which model wins on this
small dataset, but that the **same workflow** evaluated all of them -- the Experiment
Line made the comparison trivial and fair.

Other learners follow exactly the same pattern: `ClaRpart`, `ClaMultinom` and
`ClaGlmnet`.

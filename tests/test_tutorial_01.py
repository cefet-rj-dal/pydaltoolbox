"""Integration test: Tutorial 01 - First Experiment.

Python counterpart of ``Rmd/tutorial/01-first-experiment.Rmd``. Exercises the full
Experiment Line end to end: load data, split, configure learner, fit, predict,
evaluate -- exactly the rhythm the R tutorial teaches.
"""
from pydaltoolbox import ClaDtree, SampleRandom, datasets


def test_first_experiment_end_to_end():
    # 1. load data and identify the target
    iris = datasets.iris()
    slevels = list(iris["Species"].cat.categories)
    assert slevels == ["setosa", "versicolor", "virginica"]

    # 2. train/test split
    sr = SampleRandom(seed=42)
    split = sr.train_test(iris)
    train, test = split["train"], split["test"]

    # 3. configure and fit a learner
    model = ClaDtree("Species", slevels, random_state=42).fit(train)

    # 4. predict on the feature columns, then evaluate against the true labels.
    #    predict() also tolerates the full frame, but the clean habit is to pass
    #    only the features (the target column plays no role in prediction).
    train_eval = model.evaluate(
        train["Species"], model.predict(train.drop(columns="Species"))
    )
    test_eval = model.evaluate(
        test["Species"], model.predict(test.drop(columns="Species"))
    )

    assert train_eval["metrics"].iloc[0]["accuracy"] > 0.9
    assert test_eval["metrics"].iloc[0]["accuracy"] > 0.8

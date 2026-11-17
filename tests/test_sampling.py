"""Unit tests for sampling (mirrors R trans_sample examples)."""
from pydaltoolbox import (
    SampleRandom,
    SampleStratified,
    datasets,
    train_test_from_folds,
)


def test_train_test_split_sizes():
    iris = datasets.iris()
    tt = SampleRandom(seed=1).train_test(iris, perc=0.8)
    assert len(tt["train"]) == int(0.8 * len(iris))
    assert len(tt["train"]) + len(tt["test"]) == len(iris)


def test_k_fold_partitions_cover_dataset():
    iris = datasets.iris()
    folds = SampleRandom(seed=1).k_fold(iris, k=4)
    assert len(folds) == 4
    assert sum(len(f) for f in folds) == len(iris)


def test_train_test_from_folds_holds_out_kth():
    iris = datasets.iris()
    folds = SampleRandom(seed=1).k_fold(iris, k=5)
    split = train_test_from_folds(folds, k=1)
    assert len(split["test"]) == len(folds[0])
    assert len(split["train"]) == len(iris) - len(folds[0])


def test_stratified_train_test_preserves_proportions():
    iris = datasets.iris()
    overall = iris["Species"].value_counts(normalize=True).sort_index()
    tt = SampleStratified("Species", seed=1).train_test(iris, perc=0.8)
    train_prop = tt["train"]["Species"].value_counts(normalize=True).sort_index()
    # stratification keeps per-class proportions close to the overall distribution
    assert (abs(train_prop - overall) < 0.02).all()
    assert len(tt["train"]) + len(tt["test"]) == len(iris)


def test_stratified_k_fold_covers_dataset_with_all_classes():
    iris = datasets.iris()
    folds = SampleStratified("Species", seed=1).k_fold(iris, k=5)
    assert len(folds) == 5
    assert sum(len(f) for f in folds) == len(iris)
    for f in folds:
        assert set(f["Species"].unique()) == set(iris["Species"].cat.categories)

"""Unit tests for clustering (base + metrics + k-means)."""
import numpy as np

from pydaltoolbox import ClusterKmeans, datasets


def _iris_features():
    iris = datasets.iris()
    return iris.drop(columns="Species"), iris["Species"]


def test_kmeans_fit_and_cluster():
    x, _ = _iris_features()
    model = ClusterKmeans(k=3, seed=42).fit(x)
    labels = model.cluster(x)
    assert len(labels) == len(x)
    assert len(np.unique(labels)) == 3


def test_kmeans_fit_is_chainable():
    x, _ = _iris_features()
    model = ClusterKmeans(k=3, seed=42)
    assert model.fit(x) is model


def test_evaluate_internal_and_external_metrics():
    x, species = _iris_features()
    model = ClusterKmeans(k=3, seed=42).fit(x)
    labels = model.cluster(x)
    metrics = model.evaluate(labels, species)["metrics"]
    names = set(metrics["metric"])
    assert {"silhouette", "davies_bouldin", "calinski_harabasz"} <= names  # internal
    assert {"entropy", "purity", "adjusted_rand_index"} <= names  # external
    # k-means recovers the iris species reasonably well
    ari = metrics.loc[metrics["metric"] == "adjusted_rand_index", "value"].iloc[0]
    assert ari > 0.5


def test_evaluate_internal_only_without_attribute():
    x, _ = _iris_features()
    model = ClusterKmeans(k=3, seed=42).fit(x)
    metrics = model.evaluate(model.cluster(x))["metrics"]
    assert set(metrics["type"]) == {"internal"}


def test_action_dispatches_to_cluster():
    x, _ = _iris_features()
    model = ClusterKmeans(k=3, seed=42).fit(x)
    assert np.array_equal(model.action(x), model.cluster(x))

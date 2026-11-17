"""Unit tests for DBSCAN, GMM and hierarchical clustering."""
import numpy as np

from pydaltoolbox import ClusterDbscan, ClusterGmm, ClusterHclust, datasets


def _iris():
    iris = datasets.iris()
    return iris.drop(columns="Species"), iris["Species"]


def test_dbscan_fit_cluster_and_noise_metric():
    x, species = _iris()
    model = ClusterDbscan(min_pts=5).fit(x)
    labels = model.cluster(x)
    assert len(labels) == len(x)
    assert model.eps is not None and model.eps > 0  # eps estimated from kNN elbow
    metrics = model.evaluate(labels, species)["metrics"]
    assert "noise_points" in set(metrics["metric"])


def test_gmm_auto_selects_components_via_bic():
    x, species = _iris()
    model = ClusterGmm(seed=42).fit(x)
    labels = model.cluster(x)
    assert 1 <= model.g <= 10
    assert len(np.unique(labels)) == model.g
    metrics = model.evaluate(labels, species)["metrics"]
    assert "loglik" in set(metrics["metric"])


def test_gmm_fixed_components_and_predicts_new_data():
    x, _ = _iris()
    model = ClusterGmm(g=3, seed=42).fit(x)
    labels = model.cluster(x.head(10))  # GMM can assign unseen points
    assert len(labels) == 10


def test_hclust_cuts_to_k_clusters():
    x, species = _iris()
    model = ClusterHclust(k=3).fit(x)
    labels = model.cluster(x)
    assert len(np.unique(labels)) == 3
    ari = (
        model.evaluate(labels, species)["metrics"]
        .set_index("metric")
        .loc["adjusted_rand_index", "value"]
    )
    assert ari > 0.5  # hierarchical recovers species reasonably


def test_hclust_distance_threshold():
    x, _ = _iris()
    model = ClusterHclust(h=10.0, k=2).fit(x)
    labels = model.cluster(x)
    assert len(labels) == len(x)

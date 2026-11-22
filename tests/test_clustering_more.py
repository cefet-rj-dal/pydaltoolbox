"""Unit tests for DBSCAN, GMM, hierarchical, PAM and fuzzy c-means clustering."""
import numpy as np
import pandas as pd
import pytest

from pydaltoolbox import (
    ClusterCmeans,
    ClusterDbscan,
    ClusterGmm,
    ClusterHclust,
    ClusterPam,
    datasets,
)


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


def test_pam_recovers_species_and_assigns_new_data():
    x, species = _iris()
    model = ClusterPam(k=3).fit(x)
    labels = model.cluster(x)
    assert len(np.unique(labels)) == 3
    assert model.medoids_.shape == (3, x.shape[1])
    ari = (
        model.evaluate(labels, species)["metrics"]
        .set_index("metric")
        .loc["adjusted_rand_index", "value"]
    )
    assert ari > 0.5  # PAM recovers species reasonably
    # new (unseen) points are assigned to the nearest medoid
    new_labels = model.cluster(x.head(10))
    assert len(new_labels) == 10
    assert set(np.unique(new_labels)).issubset({0, 1, 2})


def test_pam_fastpam_matches_bruteforce_swap():
    """FastPAM1 must reach the same medoids/labels as the classic brute-force SWAP."""
    from pydaltoolbox.clustering.pam import _build, _euclidean_matrix, _pam

    x, _ = _iris()
    xm = x.to_numpy(dtype=float)
    d = _euclidean_matrix(xm, xm)
    k = 3

    # brute-force PAM SWAP from the same BUILD seed (reference implementation)
    medoids = _build(d, k)
    n = d.shape[0]
    for _ in range(100):
        current = d[:, medoids].min(axis=1).sum()
        best_delta, best_swap = 0.0, None
        non_medoids = [h for h in range(n) if h not in set(medoids.tolist())]
        for mi in range(k):
            for h in non_medoids:
                trial = medoids.copy()
                trial[mi] = h
                delta = d[:, trial].min(axis=1).sum() - current
                if delta < best_delta - 1e-9:
                    best_delta, best_swap = delta, (mi, h)
        if best_swap is None:
            break
        mi, h = best_swap
        medoids[mi] = h
        medoids = np.sort(medoids)

    fast = _pam(d, k, max_iter=100)
    assert set(fast.tolist()) == set(medoids.tolist())
    # identical medoids => identical total cost
    assert np.isclose(d[:, fast].min(axis=1).sum(), d[:, medoids].min(axis=1).sum())


def test_cmeans_membership_and_withinerror():
    x, species = _iris()
    model = ClusterCmeans(centers=3, m=2, seed=42).fit(x)
    labels = model.cluster(x)
    assert len(np.unique(labels)) == 3
    assert model.membership_.shape == (len(x), 3)
    # memberships are a valid soft partition (rows sum to 1)
    assert np.allclose(model.membership_.sum(axis=1), 1.0)
    metrics = model.evaluate(labels, species)["metrics"]
    assert "withinerror" in set(metrics["metric"])
    assert model.withinerror > 0


def test_cmeans_nstart_is_not_worse_on_hard_data():
    """Robust init: nstart restarts reach a within-error no worse than single-start."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(-6, 6, (8, 2))
    x = pd.DataFrame(
        np.vstack([rng.normal(c, 1.6, (60, 2)) for c in centers]), columns=["x1", "x2"]
    )
    seeds = range(8)
    mean1 = np.mean([ClusterCmeans(centers=8, m=2, nstart=1, seed=s).fit(x).withinerror for s in seeds])
    mean10 = np.mean([ClusterCmeans(centers=8, m=2, nstart=10, seed=s).fit(x).withinerror for s in seeds])
    assert mean10 <= mean1 + 1e-6  # best-of-restarts never worse on average


def test_cmeans_rejects_new_data():
    x, _ = _iris()
    model = ClusterCmeans(centers=3, seed=42).fit(x)
    with pytest.raises(ValueError):
        model.cluster(x.head(10))

"""Quality / R-parity guards for clusterers and the elbow detector (step 0.23.x).

Paired against R on the IDENTICAL iris rows (see
``daltoolbox/develop/parity/clustering_iris*``), measuring external agreement of the
partition with the reference labels (purity).

Two gaps motivated these guards:
- DBSCAN estimated ``eps`` from a wiggly GCV spline whose curvature maximum sat near the
  tail (eps too large, purity ~0.69). Reproducing R's stiff ``smooth.spline(df=2)`` via
  the Reinsch formulation moved the elbow to the real knee (purity ~0.92). sklearn's
  self-including k-NN also shifted the k-distance curve by one neighbour.
- R k-means used ``nstart=1`` (fixed on the R side); Python already restarts (n_init).

Thresholds are loose on purpose -- the point is to catch a learner or the elbow detector
collapsing, not to pin exact numbers.
"""
import warnings

import numpy as np

import pydaltoolbox as dal
from pydaltoolbox.transform import FitCurvatureMax

warnings.filterwarnings("ignore")


def _iris_xy():
    iris = dal.datasets.iris()
    return iris.drop(columns="Species"), iris["Species"]


def _purity(model):
    from pydaltoolbox.clustering.metrics import metric_purity

    x, ref = _iris_xy()
    model.fit(x)
    clu = model.cluster(x)
    return float(metric_purity(cluster=clu, attribute=ref)["value"])


def test_dbscan_eps_elbow_not_degenerate():
    """DBSCAN auto-eps must find the knee (purity ~0.92), not over-merge (~0.69)."""
    assert _purity(dal.ClusterDbscan(min_pts=3)) > 0.85


def test_partitional_clusterers_recover_structure():
    """Partitional/model-based clusterers should align well with the 3 iris species."""
    for model in (
        dal.ClusterKmeans(k=3, seed=1),
        dal.ClusterPam(k=3),
        dal.ClusterCmeans(centers=3, seed=1),
        dal.ClusterGmm(g=3, seed=1),
        dal.ClusterHclust(k=3),
    ):
        pur = _purity(model)
        assert pur > 0.80, f"{type(model).__name__} degraded: purity={pur:.3f}"


def test_fit_curvature_max_matches_stiff_spline_knee():
    """FitCurvatureMax (R smooth.spline df=2 parity) picks the knee, not the tail.

    On the iris k-distance curve the knee sits early (eps ~0.37); the old GCV spline put
    the maximum near the end (eps ~0.60). Guard the Reinsch df=2 behaviour.
    """
    from sklearn.neighbors import NearestNeighbors

    x, _ = _iris_xy()
    arr = x.to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=4).fit(arr)
    dist, _ = nn.kneighbors(arr)
    kdist = np.sort(dist[:, -1])
    res = FitCurvatureMax().transform(kdist)
    eps = float(res["y"].iloc[0])
    pos = int(res["x"].iloc[0])
    assert 0.30 < eps < 0.45, f"elbow eps off the knee: {eps:.3f}"
    assert 80 < pos < 115, f"elbow index off the knee: {pos}"

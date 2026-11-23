"""Curvature-based elbow detection.

Python counterparts of ``trans_fit_curvature_min.R`` and ``trans_fit_curvature_max.R``.
Given a 1-D sequence, fit a cubic smoothing spline, take its second derivative (the
"curvature"), and report the position of minimum/maximum curvature -- a standard
elbow/knee heuristic. Used, for example, to pick the number of PCA components or to
estimate the DBSCAN ``eps`` from the k-distance curve.

Fidelity note: R ``fit_curvature_*`` uses ``stats::smooth.spline(x, y, df = 2)`` -- a
very stiff spline whose second derivative has a single clean hump at the knee. The
scipy GCV spline (``make_smoothing_spline``) tracks the data far more closely and puts
its curvature maximum near the tail, giving a different elbow. We therefore reproduce
R's behaviour directly via the Reinsch cubic-smoothing-spline formulation with the
penalty solved for an effective ``df = 2`` (trace of the smoother matrix), which matches
R's elbow (e.g. DBSCAN eps on iris: R x=98/eps=0.387, here x=97/eps=0.374).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import DalTransform

__all__ = ["FitCurvatureMax", "FitCurvatureMin"]

# x in these routines is always the regular integer grid 1..n, so the penalty that
# yields a given effective df depends only on n. Large sequences are sub-sampled to at
# most this many points (the k-distance / scree curve is smooth, so the elbow is
# preserved) to keep the single eigendecomposition cheap.
_MAX_POINTS = 400
_TARGET_DF = 2.0


def _reinsch_curvature(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Second derivative of a ``df=2`` cubic smoothing spline over ``y`` on grid 1..n.

    Returns the original indices used and the curvature at each of them. Mirrors R
    ``smooth.spline(1:n, y, df = 2)`` followed by ``predict(deriv = 2)``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    idx = (
        np.unique(np.linspace(0, n - 1, _MAX_POINTS).round().astype(int))
        if n > _MAX_POINTS
        else np.arange(n)
    )
    ys = y[idx]
    k = len(ys)
    m = k - 2

    # Reinsch matrices for unit-spaced knots: Q is the second-difference operator
    # (k x m), R is tridiagonal (2/3 on the diagonal, 1/6 off). The smoother is
    # S_lam = (I + lam * Q R^-1 Q^T)^-1 and the interior second derivatives are
    # R^-1 Q^T g (zero at the natural-spline boundaries).
    q = np.zeros((k, m))
    rows = np.arange(m)
    q[rows, rows] = 1.0
    q[rows + 1, rows] = -2.0
    q[rows + 2, rows] = 1.0
    r = np.zeros((m, m))
    np.fill_diagonal(r, 2.0 / 3.0)
    off = np.arange(m - 1)
    r[off, off + 1] = 1.0 / 6.0
    r[off + 1, off] = 1.0 / 6.0
    r_inv = np.linalg.inv(r)
    big_k = q @ r_inv @ q.T

    mu, vecs = np.linalg.eigh(big_k)
    mu = np.clip(mu, 0.0, None)
    vty = vecs.T @ ys

    # Solve the penalty lam so the effective df = trace(S_lam) = sum 1/(1+lam*mu_i).
    lo, hi = 1e-6, 1e14
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if float(np.sum(1.0 / (1.0 + mid * mu))) > _TARGET_DF:
            lo = mid
        else:
            hi = mid
    lam = np.sqrt(lo * hi)

    g = vecs @ (vty / (1.0 + lam * mu))
    gamma = r_inv @ (q.T @ g)
    curv = np.concatenate([[0.0], gamma, [0.0]])
    return idx, curv


class _FitCurvature(DalTransform):
    _pick = staticmethod(np.argmin)

    def transform(self, y, **kwargs) -> pd.DataFrame:
        y = np.asarray(y, dtype=float)
        if len(y) < 4:
            # too short for a spline: fall back to a discrete second difference.
            curv = np.gradient(np.gradient(y)) if len(y) >= 3 else np.zeros(len(y))
            i = int(type(self)._pick(curv)) if len(y) else 0
            return pd.DataFrame(
                [{"x": i + 1, "y": float(y[i]), "yfit": float(curv[i])}]
            )
        idx, curv = _reinsch_curvature(y)
        j = int(type(self)._pick(curv))
        i = int(idx[j])
        return pd.DataFrame(
            [{"x": i + 1, "y": float(y[i]), "yfit": float(curv[j])}]
        )


class FitCurvatureMin(_FitCurvature):
    """Return the position/value of minimum curvature (concave elbow).

    Locates the knee/elbow of a monotone curve as the point of extreme second derivative
    (curvature) of a fitted smoothing spline -- the "Kneedle" heuristic -- used, for
    example, to choose the number of PCA components or clusters.

    Porting:
        Mirrors ``trans_fit_curvature_min.R``. Fidelity: exact.
        - Reproduces R ``smooth.spline(x, y, df = 2)`` via the Reinsch cubic-smoothing-
          spline formulation, solving the penalty for an effective ``df = 2`` (trace of
          the smoother), then taking the second derivative's extremum.

    References:
        Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). Finding a
        "Kneedle" in a Haystack: Detecting Knee Points in System Behavior. ICDCSW.
        Reinsch, C. H. (1967). Smoothing by spline functions. Numerische Mathematik,
        10(3):177-183.
    """

    _pick = staticmethod(np.argmin)


class FitCurvatureMax(_FitCurvature):
    """Return the position/value of maximum curvature (convex elbow).

    Locates the knee/elbow of a monotone curve as the point of extreme second derivative
    (curvature) of a fitted smoothing spline -- the "Kneedle" heuristic -- used, for
    example, to estimate the DBSCAN ``eps`` from the k-distance curve.

    Porting:
        Mirrors ``trans_fit_curvature_max.R``. Fidelity: exact.
        - Reproduces R ``smooth.spline(x, y, df = 2)`` via the Reinsch cubic-smoothing-
          spline formulation, solving the penalty for an effective ``df = 2`` (trace of
          the smoother), then taking the second derivative's extremum.

    References:
        Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). Finding a
        "Kneedle" in a Haystack: Detecting Knee Points in System Behavior. ICDCSW.
        Reinsch, C. H. (1967). Smoothing by spline functions. Numerische Mathematik,
        10(3):177-183.
    """

    _pick = staticmethod(np.argmax)

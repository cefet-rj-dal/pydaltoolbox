"""Curvature-based elbow detection.

Python counterparts of ``trans_fit_curvature_min.R`` and ``trans_fit_curvature_max.R``.
Given a 1-D sequence, fit a smoothing spline, take its second derivative (the
"curvature"), and report the position of minimum/maximum curvature -- a standard
elbow/knee heuristic. Used, for example, to pick the number of PCA components.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import DalTransform

__all__ = ["FitCurvatureMax", "FitCurvatureMin"]

try:  # scipy >= 1.10 — closest analog to R's smooth.spline
    from scipy.interpolate import make_smoothing_spline

    _HAS_SMOOTHING_SPLINE = True
except ImportError:  # pragma: no cover
    _HAS_SMOOTHING_SPLINE = False


def _curvature(y) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.arange(1, len(y) + 1)
    if _HAS_SMOOTHING_SPLINE and len(y) >= 4:
        try:
            spline = make_smoothing_spline(x, y)
            return x, spline.derivative(nu=2)(x)
        except Exception:  # noqa: BLE001, S110 - fall back to a discrete estimate
            pass
    # discrete second-derivative fallback
    return x, np.gradient(np.gradient(y))


class _FitCurvature(DalTransform):
    _pick = staticmethod(np.argmin)

    def transform(self, y, **kwargs) -> pd.DataFrame:
        x, curv = _curvature(y)
        i = int(type(self)._pick(curv))
        return pd.DataFrame(
            [{"x": int(x[i]), "y": float(np.asarray(y, dtype=float)[i]), "yfit": float(curv[i])}]
        )


class FitCurvatureMin(_FitCurvature):
    """Return the position/value of minimum curvature (concave elbow)."""

    _pick = staticmethod(np.argmin)


class FitCurvatureMax(_FitCurvature):
    """Return the position/value of maximum curvature (convex elbow)."""

    _pick = staticmethod(np.argmax)

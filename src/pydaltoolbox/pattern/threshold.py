"""Support/confidence threshold strategies (``pat_threshold.R`` analog).

These configure how :class:`~pydaltoolbox.PatApriori` estimates a minimum support (or
confidence) threshold during ``fit`` when the corresponding parameter is left at ``0``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..transform.curvature import FitCurvatureMax

__all__ = ["ConfidenceThreshold", "SupportThreshold", "item_frequency"]


def item_frequency(data: pd.DataFrame) -> pd.Series:
    """Fraction of transactions containing each item (one-hot column means)."""
    return data.astype(bool).mean(axis=0)


def _curvature_support(freq: np.ndarray) -> float:
    """Support at the knee of the sorted item-frequency curve (max curvature)."""
    freq = np.sort(np.asarray(freq, dtype=float))[::-1]
    freq = freq[np.isfinite(freq) & (freq > 0)]
    if len(freq) == 0:
        return 0.001
    if len(freq) < 4:
        return float(np.median(freq))
    return float(FitCurvatureMax().transform(freq)["y"].iloc[0])


class SupportThreshold:
    """Strategy to estimate a minimum support threshold from the data.

    Methods: ``"curvature"`` (knee of the sorted item-frequency curve, via
    :class:`~pydaltoolbox.FitCurvatureMax`), ``"min_count"`` (``min_count``/n),
    ``"sqrt"`` (``sqrt(n)``/n), ``"quantile"`` (the ``q`` quantile of item
    frequencies times ``scale``) and ``"density"`` (mean item frequency times ``scale``).
    The result is clamped to ``[min_support, max_support]``.

    Porting:
        Mirrors ``pat_support_threshold`` in ``pat_threshold.R``. Fidelity: approximate.
        - ``min``/``max`` are named ``min_support``/``max_support`` (idiomatic Python).
    """

    def __init__(
        self,
        method: str = "curvature",
        min_count: int = 10,
        q: float = 0.25,
        scale: float = 1.0,
        min_support: float = 0.001,
        max_support: float = 0.5,
    ) -> None:
        if method not in ("curvature", "min_count", "sqrt", "quantile", "density"):
            raise ValueError(
                "SupportThreshold: method must be one of 'curvature', 'min_count', "
                "'sqrt', 'quantile', 'density'."
            )
        self.method = method
        self.min_count = min_count
        self.q = q
        self.scale = scale
        self.min_support = min_support
        self.max_support = max_support

    def resolve(self, data: pd.DataFrame) -> float:
        n = len(data)
        if n <= 0:
            raise ValueError("SupportThreshold: support cannot be estimated from empty data.")
        freq = item_frequency(data)
        if self.method == "curvature":
            support = _curvature_support(freq.to_numpy())
        elif self.method == "min_count":
            support = self.min_count / n
        elif self.method == "sqrt":
            support = np.sqrt(n) / n
        elif self.method == "quantile":
            support = (
                self.min_support if freq.empty
                else float(np.quantile(freq.to_numpy(), self.q)) * self.scale
            )
        else:  # density
            support = self.min_support if freq.empty else float(freq.mean()) * self.scale
        return float(np.clip(support, self.min_support, self.max_support))


class ConfidenceThreshold:
    """Strategy to estimate a minimum confidence threshold from the data.

    Methods: ``"rhs_baseline"`` (the highest frequency among the right-hand-side items plus
    ``margin``), ``"support_adaptive"`` (``(1 - support) * scale``) and ``"fixed"``
    (``value``). The result is clamped to ``[min_confidence, max_confidence]``.

    Porting:
        Mirrors ``pat_confidence_threshold`` in ``pat_threshold.R``. Fidelity: approximate.
        - ``min``/``max`` are named ``min_confidence``/``max_confidence``.
    """

    def __init__(
        self,
        method: str = "rhs_baseline",
        value: float = 0.8,
        margin: float = 0.1,
        scale: float = 1.0,
        min_confidence: float = 0.5,
        max_confidence: float = 0.95,
    ) -> None:
        if method not in ("rhs_baseline", "support_adaptive", "fixed"):
            raise ValueError(
                "ConfidenceThreshold: method must be 'rhs_baseline', 'support_adaptive', "
                "or 'fixed'."
            )
        self.method = method
        self.value = value
        self.margin = margin
        self.scale = scale
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def resolve(self, data: pd.DataFrame, rhs=None, support: float | None = None) -> float:
        if self.method == "fixed":
            confidence = self.value
        elif self.method == "rhs_baseline":
            confidence = self._rhs_baseline(data, rhs) + self.margin
        else:  # support_adaptive
            confidence = (1.0 - (support or 0.0)) * self.scale
        return float(np.clip(confidence, self.min_confidence, self.max_confidence))

    @staticmethod
    def _rhs_baseline(data: pd.DataFrame, rhs) -> float:
        if not rhs:
            return 0.0
        freq = item_frequency(data)
        hits = [float(freq[c]) for c in rhs if c in freq.index]
        return max(hits) if hits else 0.0

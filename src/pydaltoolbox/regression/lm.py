"""Linear regression (``reg_lm.R`` analog)."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression

from .base import SklearnRegression

__all__ = ["RegLm"]


class RegLm(SklearnRegression):
    """Ordinary least squares linear regression (wraps scikit-learn)."""

    def __init__(self, attribute: str) -> None:
        super().__init__(attribute)

    def _make_model(self) -> LinearRegression:
        return LinearRegression()

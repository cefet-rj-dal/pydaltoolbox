"""Linear regression (``reg_lm.R`` analog)."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression

from .base import SklearnRegression

__all__ = ["RegLm"]


class RegLm(SklearnRegression):
    """Ordinary least squares linear regression (wraps scikit-learn).

    Models the target as a linear combination of the predictors, fitting the
    coefficients that minimize the sum of squared residuals (ordinary least squares). It
    is simple, interpretable, and the standard baseline for regression.

    References:
        Draper, N. R., and Smith, H. (1998). Applied Regression Analysis (3rd ed.).
        Wiley.
        Weisberg, S. (2005). Applied Linear Regression (3rd ed.). Wiley.
    """

    def __init__(self, attribute: str) -> None:
        super().__init__(attribute)

    def _make_model(self) -> LinearRegression:
        return LinearRegression()

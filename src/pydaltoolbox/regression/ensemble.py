"""Random forest regression (``reg_rf.R`` analog)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from .base import SklearnRegression

__all__ = ["RegRf"]


class RegRf(SklearnRegression):
    """Random forest for regression (wraps scikit-learn)."""

    def __init__(
        self,
        attribute: str,
        ntree: int = 100,
        mtry: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute)
        self.ntree = ntree
        self.mtry = mtry
        self.random_state = random_state

    def _make_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.ntree,
            max_features=self.mtry,
            random_state=self.random_state,
        )

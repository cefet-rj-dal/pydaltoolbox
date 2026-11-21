"""Random forest regression (``reg_rf.R`` analog)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from .base import SklearnRegression

__all__ = ["RegRf"]


class RegRf(SklearnRegression):
    """Random forest for regression (wraps scikit-learn).

    Averages the predictions of many decorrelated regression trees -- each grown on a
    bootstrap sample using a random subset of features at every split -- which reduces
    variance and makes the model robust to overfitting on tabular data. Key
    hyperparameters are the number of trees (``ntree``) and the number of variables tried
    per split (``mtry``).

    References:
        Breiman, L. (2001). Random Forests. Machine Learning, 45(1):5-32.
        Liaw, A., and Wiener, M. (2002). Classification and Regression by randomForest.
        R News, 2(3):18-22.
    """

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

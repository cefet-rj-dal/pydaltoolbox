"""Support Vector Machine classifier (``cla_svm.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.svm import SVC

from .base import SklearnClassification

__all__ = ["ClaSvm"]


class ClaSvm(SklearnClassification):
    """Support vector classifier (wraps scikit-learn's SVC).

    ``probability=True`` is required so ``predict`` can return class scores.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        kernel: str = "rbf",
        cost: float = 1.0,
        gamma: str | float = "scale",
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.kernel = kernel
        self.cost = cost
        self.gamma = gamma
        self.random_state = random_state

    def _make_model(self) -> SVC:
        return SVC(
            kernel=self.kernel,
            C=self.cost,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state,
        )

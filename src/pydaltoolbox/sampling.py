"""Data sampling abstractions.

Python counterpart of ``trans_sample.R`` and ``trans_sample_random.R``. Sampling
strategies are transforms that produce train/test splits and k-fold partitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .adjust import adjust_dataframe
from .base import DalTransform

__all__ = ["DataSample", "SampleRandom", "SampleStratified", "train_test_from_folds"]


class DataSample(DalTransform):
    """Base class for sampling strategies."""

    def train_test(self, data, perc: float = 0.8, **kwargs) -> dict:
        """Return ``{'train': ..., 'test': ...}``. Default is empty."""
        return {}

    def k_fold(self, data, k: int) -> list[pd.DataFrame]:
        """Return a list of ``k`` folds. Default is empty."""
        return []


class SampleRandom(DataSample):
    """Train/test split and k-fold partitioning by simple random sampling.

    Parameters
    ----------
    seed:
        Optional seed for reproducible splits (Python/R differ in RNG, so splits
        will not match R exactly, only be reproducible within Python).
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def train_test(self, data, perc: float = 0.8, **kwargs) -> dict:
        data = adjust_dataframe(data)
        n = len(data)
        n_train = int(perc * n)
        idx = self._rng().permutation(n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        train = data.iloc[np.sort(train_idx)].reset_index(drop=True)
        test = data.iloc[np.sort(test_idx)].reset_index(drop=True)
        return {"train": train, "test": test}

    def k_fold(self, data, k: int) -> list[pd.DataFrame]:
        data = adjust_dataframe(data)
        idx = self._rng().permutation(len(data))
        splits = np.array_split(idx, k)
        return [data.iloc[np.sort(s)].reset_index(drop=True) for s in splits]


class SampleStratified(SampleRandom):
    """Train/test split and k-fold that preserve the target class proportions.

    Use this sampler when the response distribution matters, especially in
    imbalanced classification. Mirrors R's ``sample_stratified(attribute)``.

    Parameters
    ----------
    attribute:
        Name of the target column whose class proportions should be preserved.
    seed:
        Optional seed for reproducible splits.
    """

    def __init__(self, attribute: str, seed: int | None = None) -> None:
        super().__init__(seed)
        self.attribute = attribute

    def train_test(self, data, perc: float = 0.8, **kwargs) -> dict:
        data = adjust_dataframe(data)
        train, test = train_test_split(
            data,
            train_size=perc,
            stratify=data[self.attribute],
            random_state=self.seed,
        )
        return {
            "train": train.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }

    def k_fold(self, data, k: int) -> list[pd.DataFrame]:
        data = adjust_dataframe(data)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        return [
            data.iloc[test_idx].reset_index(drop=True)
            for _, test_idx in skf.split(data, data[self.attribute])
        ]


def train_test_from_folds(folds: list[pd.DataFrame], k: int) -> dict:
    """Build one train/test split from a list of folds.

    The ``k``-th fold (1-based, as in R) is the test set; the remaining folds are
    concatenated into the training set.
    """
    test = folds[k - 1]
    train_parts = [f for i, f in enumerate(folds, start=1) if i != k]
    train = pd.concat(train_parts, ignore_index=True) if train_parts else test.iloc[0:0]
    return {"train": train, "test": test}

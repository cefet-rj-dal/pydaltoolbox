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

__all__ = [
    "DataSample",
    "SampleGroups",
    "SampleRandom",
    "SampleSimple",
    "SampleStratified",
    "train_test_from_folds",
]


class DataSample(DalTransform):
    """Base class for sampling strategies.

    Sampling methods partition data for honest model evaluation: a held-out test set
    estimates generalization, and k-fold cross-validation averages performance over
    several train/test rotations to reduce the variance of that estimate. Subclasses
    implement :meth:`train_test` and :meth:`k_fold`.

    Porting:
        Mirrors ``trans_sample.R``. Fidelity: exact.
        - Base class: defines the ``train_test``/``k_fold`` contract; subclasses sample.

    References:
        Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy
        Estimation and Model Selection. IJCAI.
    """

    def train_test(self, data, perc: float = 0.8, **kwargs) -> dict:
        """Return ``{'train': ..., 'test': ...}``. Default is empty."""
        return {}

    def k_fold(self, data, k: int) -> list[pd.DataFrame]:
        """Return a list of ``k`` folds. Default is empty."""
        return []


class SampleRandom(DataSample):
    """Train/test split and k-fold partitioning by simple random sampling.

    Partitions rows uniformly at random: ``train_test`` holds out a fraction for testing,
    and ``k_fold`` divides the data into ``k`` equal parts for cross-validation. Simple and
    unbiased when classes are reasonably balanced (use :class:`SampleStratified` otherwise).

    Porting:
        Mirrors ``trans_sample_random.R``. Fidelity: approximate.
        - numpy RNG differs from R, so splits do not match R exactly (reproducible within
          Python via ``seed``).

    Parameters
    ----------
    seed:
        Optional seed for reproducible splits (Python/R differ in RNG, so splits
        will not match R exactly, only be reproducible within Python).

    References:
        Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy
        Estimation and Model Selection. IJCAI.
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

    Stratified sampling keeps each class represented in the same proportion across the
    splits as in the full data, which lowers the variance and bias of the performance
    estimate -- important for imbalanced classification where random splits can under- or
    over-represent rare classes.

    Porting:
        Mirrors ``trans_sample_strat.R``. Fidelity: approximate.
        - scikit-learn ``train_test_split``/``StratifiedKFold`` backend; numpy/sklearn RNG
          differs from R.

    Parameters
    ----------
    attribute:
        Name of the target column whose class proportions should be preserved.
    seed:
        Optional seed for reproducible splits.

    References:
        Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy
        Estimation and Model Selection. IJCAI.
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


class SampleSimple(DalTransform):
    """Draw a simple random sample of rows, with or without replacement.

    Simple random sampling: every row has an equal (or ``prob``-weighted) chance of being
    drawn. Without replacement it yields a subset; with replacement it supports bootstrap
    resampling. Useful for subsetting large data or building resamples.

    ``transform`` returns a sampled subset of the rows (or elements, for a 1-D input).

    Porting:
        Mirrors ``trans_sample_simple.R``. Fidelity: approximate.
        - numpy/pandas RNG differs from R (reproducible within Python via ``seed``).

    References:
        Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley.

    Parameters
    ----------
    size:
        Number of samples to draw.
    replace:
        Sample with replacement if ``True``.
    prob:
        Optional per-row sampling weights.
    seed:
        Optional seed for reproducibility.
    """

    def __init__(self, size: int, replace: bool = False, prob=None, seed: int | None = None) -> None:
        self.size = size
        self.replace = replace
        self.prob = prob
        self.seed = seed

    def transform(self, data, **kwargs):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.sample(
                n=self.size, replace=self.replace, weights=self.prob, random_state=self.seed
            ).reset_index(drop=True)
        arr = np.asarray(data)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(arr), size=self.size, replace=self.replace, p=self.prob)
        return arr[idx]


class SampleGroups(DalTransform):
    """Cluster (group) sampling: keep all rows of ``n_groups`` sampled groups.

    Cluster sampling draws whole pre-existing groups at random and keeps all their rows,
    rather than sampling individual rows. It is the right choice when observations are
    naturally nested (e.g. by subject, site or batch) and must be kept together to avoid
    leakage between train and test.

    Groups are the distinct values of ``attribute`` already present in the data (not
    inferred by clustering).

    Porting:
        Mirrors ``trans_sample_groups.R``. Fidelity: approximate.
        - numpy RNG differs from R (reproducible within Python via ``seed``).

    References:
        Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley.
    """

    def __init__(self, attribute: str, n_groups: int, seed: int | None = None) -> None:
        self.attribute = attribute
        self.n_groups = n_groups
        self.seed = seed

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        if self.attribute not in data.columns:
            raise ValueError(f"sample_groups: '{self.attribute}' must be a valid column name.")
        groups = pd.unique(data[self.attribute])
        if len(groups) < self.n_groups:
            raise ValueError("sample_groups: n_groups exceeds the number of available groups.")
        rng = np.random.default_rng(self.seed)
        selected = rng.choice(groups, size=self.n_groups, replace=False)
        return data[data[self.attribute].isin(selected)].reset_index(drop=True)

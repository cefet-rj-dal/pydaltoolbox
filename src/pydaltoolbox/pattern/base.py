"""Pattern mining base (``pat_pattern_miner.R`` and ``pat_utils.R`` analogs).

Pattern miners follow a lightweight Experiment Line: :meth:`PatternMiner.fit`
validates the mining input and stores a schema signature, :meth:`PatternMiner.discover`
(reached through ``action``) runs the mining algorithm on data compatible with that
schema, and :meth:`PatternMiner.evaluate` summarizes pattern quality and the effect of
any filtering.

Transactions are represented as a one-hot encoded :class:`pandas.DataFrame`: one row per
transaction, one boolean column per item. Discovered patterns are returned as a
:class:`pandas.DataFrame` (``support``/``itemsets`` for itemsets; antecedents,
consequents and interest measures for rules), carrying the pre-filter count in
``patterns.attrs['filtered_from']`` so the retained-ratio metric can be computed.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ..base import DalLearner

__all__ = ["PatUtils", "PatternMiner", "pattern_items", "pattern_length"]


def _require_mlxtend():
    """Import mlxtend lazily, with a clear install hint when it is missing."""
    try:
        from mlxtend import frequent_patterns
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Pattern mining requires the optional 'mlxtend' dependency. "
            "Install it with: pip install pydaltoolbox[patterns]"
        ) from exc
    return frequent_patterns


def pattern_items(patterns: pd.DataFrame) -> pd.Series:
    """Set of items appearing in each pattern (itemset, or rule antecedents+consequents)."""
    if "itemsets" in patterns.columns:
        return patterns["itemsets"].apply(lambda s: frozenset(s))
    if "antecedents" in patterns.columns and "consequents" in patterns.columns:
        return patterns.apply(
            lambda r: frozenset(r["antecedents"]) | frozenset(r["consequents"]), axis=1
        )
    return pd.Series([frozenset()] * len(patterns), index=patterns.index)


def pattern_length(patterns: pd.DataFrame) -> pd.Series:
    """Number of items in each pattern."""
    return pattern_items(patterns).apply(len)


def _metric(metric: str, value, type_: str = "intrinsic") -> dict:
    value = float("nan") if pd.isna(value) else float(value)
    return {"metric": metric, "value": value, "type": type_}


def metric_pattern_count(patterns: pd.DataFrame, obj=None) -> dict:
    return _metric("pattern_count", len(patterns), "intrinsic")


def _mean_column(patterns: pd.DataFrame, column: str, name: str) -> dict:
    value = patterns[column].mean() if column in patterns.columns and len(patterns) else float("nan")
    return _metric(name, value, "intrinsic")


def metric_mean_support(patterns: pd.DataFrame, obj=None) -> dict:
    return _mean_column(patterns, "support", "mean_support")


def metric_mean_confidence(patterns: pd.DataFrame, obj=None) -> dict:
    return _mean_column(patterns, "confidence", "mean_confidence")


def metric_mean_lift(patterns: pd.DataFrame, obj=None) -> dict:
    return _mean_column(patterns, "lift", "mean_lift")


def metric_mean_length(patterns: pd.DataFrame, obj=None) -> dict:
    value = pattern_length(patterns).mean() if len(patterns) else float("nan")
    return _metric("mean_length", value, "intrinsic")


def metric_retained_ratio(patterns: pd.DataFrame, obj=None) -> dict:
    total = patterns.attrs.get("filtered_from")
    if not total:
        return _metric("retained_ratio", float("nan"), "filter")
    return _metric("retained_ratio", len(patterns) / total, "filter")


class PatUtils:
    """Quality-filter builders and descriptive metrics for pattern mining.

    Groups two kinds of helpers: quality-filter builders (:meth:`quality_min` and
    :meth:`quality_max`, which return threshold dictionaries consumed by a miner's
    ``quality_filter``) and descriptive metric callables (pattern count, mean support,
    mean confidence, mean lift, mean length, and the retained ratio after filtering).

    Porting:
        Mirrors ``pat_utils.R``. Fidelity: adapted.
        - Idiomatic-Python API (see :class:`~pydaltoolbox.base.DalBase`); metrics operate
          on a patterns :class:`pandas.DataFrame` instead of an ``arules`` object.
    """

    @staticmethod
    def quality_min(support=None, confidence=None, lift=None) -> dict:
        """Lower-bound quality filter; keeps patterns at or above each given threshold."""
        bounds = {k: v for k, v in (("support", support), ("confidence", confidence),
                                    ("lift", lift)) if v is not None}
        return {"min": bounds}

    @staticmethod
    def quality_max(support=None, confidence=None, lift=None) -> dict:
        """Upper-bound quality filter; keeps patterns at or below each given threshold."""
        bounds = {k: v for k, v in (("support", support), ("confidence", confidence),
                                    ("lift", lift)) if v is not None}
        return {"max": bounds}

    metric_pattern_count = staticmethod(metric_pattern_count)
    metric_mean_support = staticmethod(metric_mean_support)
    metric_mean_confidence = staticmethod(metric_mean_confidence)
    metric_mean_lift = staticmethod(metric_mean_lift)
    metric_mean_length = staticmethod(metric_mean_length)
    metric_retained_ratio = staticmethod(metric_retained_ratio)


class PatternMiner(DalLearner):
    """Base class for frequent pattern and sequence mining.

    Pattern miners follow a lightweight Experiment Line: :meth:`fit` validates the mining
    input and stores a schema signature (the sorted item columns), :meth:`discover`
    (reached through :meth:`action`) runs the mining algorithm on data compatible with
    that schema, and :meth:`evaluate` summarizes pattern quality and the effect of any
    filtering. Transactions are a one-hot encoded :class:`pandas.DataFrame`; discovered
    patterns are a :class:`pandas.DataFrame`.

    Porting:
        Mirrors ``pat_pattern_miner.R``. Fidelity: adapted.
        - Idiomatic-Python API (see :class:`~pydaltoolbox.base.DalBase`); ``action``
          dispatches to :meth:`discover`.
        - Transactions are one-hot ``DataFrame`` columns (the ``arules`` ``transactions``
          object has no Python counterpart); the schema signature is the sorted columns.
    """

    def __init__(self) -> None:
        self.schema: list | None = None
        self.data_class: str | None = None
        self.fitted = False
        self.include: Iterable | None = None
        self.exclude: Iterable | None = None
        self.quality_filter: dict | None = None
        self.pattern_kind = "patterns"
        self.eval_metrics = [
            metric_pattern_count,
            metric_mean_support,
            metric_mean_length,
            metric_retained_ratio,
        ]

    # -- schema -----------------------------------------------------------------
    @staticmethod
    def _schema(data: pd.DataFrame) -> list:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("PatternMiner: transactions must be a one-hot pandas DataFrame.")
        return sorted(map(str, data.columns))

    @staticmethod
    def _as_onehot(data: pd.DataFrame) -> pd.DataFrame:
        """Validate and coerce a one-hot transaction matrix to a boolean DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("PatternMiner: transactions must be a one-hot pandas DataFrame.")
        values = data.to_numpy()
        if not ((values == 0) | (values == 1) | (values == True) | (values == False)).all():
            raise ValueError(
                "PatternMiner: one-hot transactions must contain only 0/1 or boolean values."
            )
        return data.astype(bool)

    def _validate_schema(self, data: pd.DataFrame) -> None:
        if self.schema is not None and self._schema(data) != self.schema:
            raise ValueError("PatternMiner: discover data schema differs from fit schema.")

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise ValueError(f"{type(self).__name__} must be fitted before discovery.")

    # -- contract ---------------------------------------------------------------
    def fit(self, data, **kwargs) -> PatternMiner:
        data = self._as_onehot(data)
        self.schema = self._schema(data)
        self.data_class = type(data).__name__
        self.fitted = True
        return self

    def discover(self, data, **kwargs) -> pd.DataFrame:  # pragma: no cover - overridden
        raise NotImplementedError

    def action(self, data, **kwargs):
        """For pattern miners, ``action`` dispatches to :meth:`discover`."""
        return self.discover(data, **kwargs)

    def evaluate(self, patterns: pd.DataFrame, **kwargs) -> dict:
        rows = [fn(patterns=patterns, obj=self) for fn in self.eval_metrics]
        return {"patterns": patterns, "metrics": pd.DataFrame(rows, columns=["metric", "value", "type"])}

    # -- filters ----------------------------------------------------------------
    def _apply_item_filter(self, patterns: pd.DataFrame) -> pd.DataFrame:
        if self.include is None and self.exclude is None:
            return patterns
        items = pattern_items(patterns)
        keep = pd.Series(True, index=patterns.index)
        if self.include is not None:
            include = set(self.include)
            keep &= items.apply(lambda s: s.issubset(include))
        if self.exclude is not None:
            exclude = set(self.exclude)
            keep &= items.apply(lambda s: s.isdisjoint(exclude))
        return patterns[keep]

    def _apply_quality_filter(self, patterns: pd.DataFrame) -> pd.DataFrame:
        flt = self.quality_filter
        if not flt:
            return patterns
        keep = pd.Series(True, index=patterns.index)
        for name, threshold in flt.get("min", {}).items():
            if name in patterns.columns:
                keep &= patterns[name] >= threshold
        for name, threshold in flt.get("max", {}).items():
            if name in patterns.columns:
                keep &= patterns[name] <= threshold
        return patterns[keep]

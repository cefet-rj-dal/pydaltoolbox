"""Association-rule filters (``pat_rule_filter.R`` analog).

A rule filter is a :class:`~pydaltoolbox.base.DalTransform` over a rules
:class:`pandas.DataFrame` (from :class:`~pydaltoolbox.PatApriori` with ``target='rules'``):
``transform`` returns a subset of the rules, and ``action`` dispatches to it.
:class:`RuleFilterNone` keeps every rule; :class:`RuleFilterDara` keeps rules whose
antecedent values are in divergence according to DARA.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalTransform
from .dara import dara, dara_score

__all__ = ["RuleFilter", "RuleFilterDara", "RuleFilterInterest", "RuleFilterNone"]


def _with_interest_measures(rules: pd.DataFrame) -> pd.DataFrame:
    """Add the ``kulc``/``ir``/``count`` columns derived from mlxtend's rule measures."""
    out = rules.copy()
    if "kulc" not in out.columns and "kulczynski" in out.columns:
        out["kulc"] = out["kulczynski"]
    if "ir" not in out.columns and {"antecedent support", "consequent support", "support"}.issubset(
        out.columns
    ):
        denom = out["antecedent support"] + out["consequent support"] - out["support"]
        out["ir"] = (out["antecedent support"] - out["consequent support"]).abs() / denom
    if "count" not in out.columns and "support" in out.columns:
        n = out.attrs.get("n_transactions")
        if n:
            out["count"] = (out["support"] * n).round().astype(int)
    return out


class RuleFilter(DalTransform):
    """Base class for association-rule filters.

    A filter is a transform over a rules DataFrame: :meth:`transform` returns a subset of
    the rules (and ``action`` dispatches to it), recording the pre-filter count in
    ``result.attrs['filtered_from']`` so the retained ratio can be measured. ``fit`` is a
    no-op (filters are stateless or configured at construction).

    Porting:
        Mirrors ``pat_rule_filter.R``. Fidelity: adapted.
        - A :class:`~pydaltoolbox.base.DalTransform` over a rules DataFrame; the R filters
          are likewise ``dal_transform`` objects (``transform(filter, rules)``), with
          ``pat_filter_rules`` kept as an equivalent entry point.
    """

    def transform(self, rules: pd.DataFrame, **kwargs) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class RuleFilterNone(RuleFilter):
    """Identity rule filter: keeps every rule unchanged.

    Porting:
        Mirrors ``pat_rule_filter_none`` in ``pat_rule_filter.R``. Fidelity: adapted.
    """

    def transform(self, rules: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = rules.copy()
        out.attrs["filtered_from"] = len(out)
        return out


class RuleFilterInterest(RuleFilter):
    """Rule filter based on classical interest measures.

    Keeps rules that pass every configured threshold: minimum lift, Kulczynski (``kulc``),
    support, confidence and count, and maximum imbalance ratio (``ir``). ``lift > 1``, high
    Kulczynski and low imbalance ratio are the usual criteria for an interesting rule
    (Han, Kamber & Pei). The ``kulc``, ``ir`` and ``count`` columns are derived from the
    measures ``mlxtend.association_rules`` already reports (``count`` needs the transaction
    count, which :class:`~pydaltoolbox.PatApriori` stores in ``rules.attrs``).

    Porting:
        Mirrors ``pat_rule_filter_interest`` in ``pat_rule_filter.R``. Fidelity: approximate.
        - Thresholds on the ``mlxtend`` rule measures instead of ``arules::interestMeasure``.

    References:
        Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques (3rd
        ed.). Morgan Kaufmann.
    """

    def __init__(
        self,
        lift_min: float | None = 1.0,
        kulc_min: float | None = None,
        ir_max: float | None = None,
        support_min: float | None = None,
        confidence_min: float | None = None,
        count_min: float | None = None,
    ) -> None:
        self.lift_min = lift_min
        self.kulc_min = kulc_min
        self.ir_max = ir_max
        self.support_min = support_min
        self.confidence_min = confidence_min
        self.count_min = count_min

    def transform(self, rules: pd.DataFrame, **kwargs) -> pd.DataFrame:
        total = len(rules)
        out = _with_interest_measures(rules)
        keep = pd.Series(True, index=out.index)
        for column, threshold, op in (
            ("lift", self.lift_min, "ge"),
            ("kulc", self.kulc_min, "ge"),
            ("ir", self.ir_max, "le"),
            ("support", self.support_min, "ge"),
            ("confidence", self.confidence_min, "ge"),
            ("count", self.count_min, "ge"),
        ):
            if threshold is None:
                continue
            if column not in out.columns:
                raise ValueError(f"RuleFilterInterest: rule measure '{column}' is unavailable.")
            keep &= out[column] >= threshold if op == "ge" else out[column] <= threshold
        out = out[keep].reset_index(drop=True)
        out.attrs["filtered_from"] = total
        return out


class RuleFilterDara(RuleFilter):
    """Divergent Association Ranking Analysis (DARA) rule filter.

    DARA assumes attribute-value independence: under it, a value that is more frequent in
    the data should appear in more association rules than a less frequent one. A value for
    which this expectation is violated is *divergent*, and rules built on divergent values
    carry the non-trivial structure DARA is meant to surface. This filter scores each rule
    by aggregating (``score`` = ``"sum"``/``"mean"``/``"max"``) the DARA divergence
    counters of its antecedent values and keeps the rules whose score reaches ``min_score``
    (and, if ``top`` is given, the ``top`` highest-scored). The kept rules gain a
    ``dara_score`` column, and the per-attribute ranking is stored in
    ``result.attrs['dara_ranking']``.

    Rules are expected as produced by :class:`~pydaltoolbox.PatApriori` (``target='rules'``)
    over transactions whose items are ``attribute=value`` labels, so the antecedent values
    can be matched against the reference ``data`` columns.

    Porting:
        Canonical implementation; the R ``pat_rule_filter.R`` (``pat_rule_filter_dara``) is
        a backport of this. Fidelity: adapted.
        - Operates on a rules DataFrame instead of an ``arules`` rules object.

    References:
        Baroni, L., Salles, R., Salles, S., Guedes, G., Porto, F., Bezerra, E., Barcellos,
        C., Pedroso, M., & Ogasawara, E. (2020). An analysis of malaria in the Brazilian
        Legal Amazon using divergent association rules. Journal of Biomedical Informatics,
        108, 103512.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        rhs_attr: str,
        rhs_value=None,
        attributes: Iterable | None = None,
        min_score: float = 1,
        top: int | None = None,
        score: str = "sum",
        use_abs: bool = False,
        rule_weight: str | None = None,
    ) -> None:
        if score not in ("sum", "mean", "max"):
            raise ValueError("RuleFilterDara: score must be 'sum', 'mean', or 'max'.")
        if rhs_attr is None:
            raise ValueError("RuleFilterDara requires 'rhs_attr'.")
        self.data = adjust_dataframe(data)
        self.rhs_attr = rhs_attr
        self.rhs_value = rhs_value
        self.attributes = attributes
        self.min_score = min_score
        self.top = top
        self.score = score
        self.use_abs = use_abs
        self.rule_weight = rule_weight

    def transform(self, rules: pd.DataFrame, **kwargs) -> pd.DataFrame:
        total = len(rules)
        ranking = dara(
            self.data, rules, rhs_attr=self.rhs_attr, rhs_value=self.rhs_value,
            attributes=self.attributes, rule_weight=self.rule_weight,
        )
        scores = np.array([
            dara_score(ranking, ante, how=self.score, use_abs=self.use_abs)
            for ante in rules["antecedents"]
        ]) if total else np.empty(0)

        keep = scores >= self.min_score
        if self.top is not None and total:
            order = np.argsort(-scores, kind="stable")
            top_keep = np.zeros(total, dtype=bool)
            top_keep[order[: self.top]] = True
            keep &= top_keep

        out = rules[keep].copy()
        out["dara_score"] = scores[keep]
        out = out.reset_index(drop=True)
        out.attrs["filtered_from"] = total
        out.attrs["dara_ranking"] = ranking
        return out

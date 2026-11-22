"""Apriori frequent itemsets and association rules (``pat_apriori.R`` analog).

Wraps ``mlxtend.frequent_patterns`` (``apriori`` and ``association_rules``) behind the
Experiment Line contract. ``target`` selects frequent itemsets or association rules;
item, length and quality filters narrow the result.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .base import (
    PatternMiner,
    _require_mlxtend,
    metric_mean_confidence,
    metric_mean_length,
    metric_mean_lift,
    metric_mean_support,
    metric_pattern_count,
    metric_retained_ratio,
    pattern_length,
)
from .rule_filter import RuleFilter, RuleFilterNone

__all__ = ["PatApriori"]


class PatApriori(PatternMiner):
    """Apriori frequent itemsets and association rules.

    Apriori mines frequent itemsets with a breadth-first, level-wise search that exploits
    the downward-closure (anti-monotone) property of support: every subset of a frequent
    itemset is itself frequent, so any candidate with an infrequent subset is pruned
    before its support is counted. From the frequent itemsets it derives association rules
    whose confidence (and other interest measures such as lift) meet a threshold. With
    ``target='frequent itemsets'`` only the itemsets are returned; with ``target='rules'``
    the rules are returned with their support, confidence and lift.

    Transactions are a one-hot encoded :class:`pandas.DataFrame`; patterns are returned as
    a :class:`pandas.DataFrame`. ``minlen``/``maxlen`` bound the itemset length,
    ``lhs``/``rhs`` constrain the items allowed on each side of a rule, ``include``/
    ``exclude`` constrain the items in any pattern, and ``quality_filter`` (from
    :class:`~pydaltoolbox.pattern.PatUtils`) applies support/confidence/lift bounds. For
    ``target='rules'``, a ``rule_filter`` (e.g. :class:`~pydaltoolbox.RuleFilterDara`)
    applies a post-mining interestingness filter to the discovered rules.

    Porting:
        Mirrors ``pat_apriori.R``. Fidelity: approximate.
        - Wraps ``mlxtend`` (``apriori``/``association_rules``) rather than ``arules``;
          the two engines report the same support, confidence and lift.
        - Support auto-estimation (``supp=0`` with a support strategy) and rule filters are
          deferred to a later step; ``supp`` (and ``conf`` for rules) are used as given.
        - ``minlen`` and the ``lhs``/``rhs`` appearance constraints are applied as a
          post-filter over the mined patterns.

    References:
        Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules in
        Large Databases. Proceedings of the 20th International Conference on Very Large
        Data Bases (VLDB), 487-499.
    """

    def __init__(
        self,
        target: str = "rules",
        supp: float = 0.1,
        conf: float = 0.8,
        minlen: int = 1,
        maxlen: int = 10,
        lhs: Iterable | None = None,
        rhs: Iterable | None = None,
        include: Iterable | None = None,
        exclude: Iterable | None = None,
        quality_filter: dict | None = None,
        rule_filter: RuleFilter | None = None,
    ) -> None:
        super().__init__()
        if target not in ("rules", "frequent itemsets"):
            raise ValueError("PatApriori: target must be 'rules' or 'frequent itemsets'.")
        if supp <= 0:
            raise ValueError("PatApriori: supp must be > 0 (support auto-estimation is deferred).")
        if rule_filter is None:
            rule_filter = RuleFilterNone()
        elif not isinstance(rule_filter, RuleFilter):
            raise TypeError("PatApriori: rule_filter must be a RuleFilter instance.")
        if target != "rules" and not isinstance(rule_filter, RuleFilterNone):
            raise ValueError("PatApriori: a rule_filter can only be used with target='rules'.")
        self.target = target
        self.supp = supp
        self.conf = conf
        self.minlen = minlen
        self.maxlen = maxlen
        self.lhs = lhs
        self.rhs = rhs
        self.include = include
        self.exclude = exclude
        self.quality_filter = quality_filter
        self.rule_filter = rule_filter
        self.pattern_kind = "rules" if target == "rules" else "itemsets"
        if target == "rules":
            self.eval_metrics = [
                metric_pattern_count,
                metric_mean_support,
                metric_mean_confidence,
                metric_mean_lift,
                metric_mean_length,
                metric_retained_ratio,
            ]
        else:
            self.eval_metrics = [
                metric_pattern_count,
                metric_mean_support,
                metric_mean_length,
                metric_retained_ratio,
            ]

    def _apply_appearance(self, rules: pd.DataFrame) -> pd.DataFrame:
        if self.lhs is None and self.rhs is None:
            return rules
        keep = pd.Series(True, index=rules.index)
        if self.lhs is not None:
            lhs = set(self.lhs)
            keep &= rules["antecedents"].apply(lambda s: frozenset(s).issubset(lhs))
        if self.rhs is not None:
            rhs = set(self.rhs)
            keep &= rules["consequents"].apply(lambda s: frozenset(s).issubset(rhs))
        return rules[keep]

    def discover(self, data, **kwargs) -> pd.DataFrame:
        self._require_fitted()
        fp = _require_mlxtend()
        data = self._as_onehot(data)
        self._validate_schema(data)

        itemsets = fp.apriori(data, min_support=self.supp, use_colnames=True, max_len=self.maxlen)

        if self.target == "rules":
            # association_rules needs the full itemset table (including singletons) to
            # resolve antecedent/consequent supports; minlen is a post-filter on rules.
            if len(itemsets):
                patterns = fp.association_rules(
                    itemsets, metric="confidence", min_threshold=self.conf
                )
            else:
                patterns = itemsets.iloc[0:0]
            patterns = self._apply_appearance(patterns)
            if self.minlen > 1 and len(patterns):
                patterns = patterns[pattern_length(patterns) >= self.minlen]
        else:
            patterns = itemsets
            if self.minlen > 1 and len(patterns):
                patterns = patterns[pattern_length(patterns) >= self.minlen]

        total = len(patterns)
        patterns = self._apply_item_filter(patterns)
        patterns = self._apply_quality_filter(patterns)

        ranking = None
        if self.target == "rules":
            patterns = self.rule_filter.action(patterns)
            ranking = patterns.attrs.get("dara_ranking")

        patterns = patterns.reset_index(drop=True)
        patterns.attrs["filtered_from"] = total
        if ranking is not None:
            patterns.attrs["dara_ranking"] = ranking
        return patterns

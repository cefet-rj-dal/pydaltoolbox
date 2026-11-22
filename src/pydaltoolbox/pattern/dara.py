"""Divergent Association Ranking Analysis (DARA) (``pat_dara.R`` analog).

DARA ranks the values of each explanatory attribute by how much their prominence in a set
of association rules diverges from their frequency in the reference data, under the
hypothesis of attribute-value independence.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from ..adjust import adjust_dataframe

__all__ = ["dara", "parse_items"]


def parse_items(itemset: Iterable) -> dict:
    """Parse ``attribute=value`` item labels into a ``{attribute: value}`` mapping.

    Each item in ``itemset`` (e.g. a rule antecedent ``frozenset``) is split on its first
    ``=``; items without ``=`` are ignored. When an attribute repeats, the last value wins.
    """
    out: dict = {}
    for item in itemset:
        label = str(item)
        if "=" in label:
            attr, _, value = label.partition("=")
            out[attr.strip()] = value.strip()
    return out


def _dara_counter(dataset: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """Pairwise divergence counter (Algorithm 2 of Baroni et al., 2020).

    For every unordered pair of values, a value that is more frequent in the data yet
    appears in fewer rules (or vice versa) is in divergence: its counter moves away from
    the other's. The result is independent of the order of the values.
    """
    n = len(dataset)
    counter = np.zeros(n)
    for j in range(n - 1):
        for k in range(j + 1, n):
            if dataset[j] > dataset[k] and rules[j] < rules[k]:
                counter[j] -= 1
                counter[k] += 1
            elif dataset[j] < dataset[k] and rules[j] > rules[k]:
                counter[j] += 1
                counter[k] -= 1
    return counter


def _rule_value_counts(parsed: list, attribute: str, weights: np.ndarray | None) -> dict:
    """Per-value rule frequency for one attribute (optionally weighted)."""
    counts: dict = {}
    for i, mapping in enumerate(parsed):
        if attribute in mapping:
            value = mapping[attribute]
            counts[value] = counts.get(value, 0.0) + (1.0 if weights is None else weights[i])
    return counts


def dara(
    data: pd.DataFrame,
    rules: pd.DataFrame,
    rhs_attr: str,
    rhs_value=None,
    attributes: Iterable | None = None,
    rule_weight: str | None = None,
) -> dict:
    """Divergent Association Ranking Analysis.

    For each explanatory attribute, compares the value-frequency ordering in a reference
    dataset with the ordering by number of rules (antecedent occurrences) in a rule set,
    and returns a per-attribute table with a divergence counter ``c``. A value more
    prominent in the rules than its data frequency would predict gets a higher ``c``; one
    less prominent gets a lower ``c``.

    Parameters
    ----------
    data : DataFrame
        Reference data (columns are attributes, values are categories).
    rules : DataFrame
        Association rules from :class:`~pydaltoolbox.PatApriori` (``target='rules'``); the
        ``antecedents`` column holds ``frozenset`` of ``attribute=value`` item labels.
    rhs_attr : str
        Right-hand-side attribute of the mined rules; excluded from the comparison.
    rhs_value : optional
        If given, restrict ``data`` to rows where ``rhs_attr == rhs_value`` before counting.
    attributes : iterable, optional
        Attributes to compare. Defaults to the attributes present both in ``data`` and in
        the rule antecedents, minus ``rhs_attr``.
    rule_weight : str, optional
        Numeric rule column used as rule frequency instead of one count per rule.

    Returns
    -------
    dict
        Maps each attribute to a DataFrame with columns ``value``, ``dataset``, ``rules``
        and ``c`` (the divergence counter), sorted by ``c`` then ``value``.

    References
    ----------
    Baroni, L., Salles, R., Salles, S., Guedes, G., Porto, F., Bezerra, E., Barcellos, C.,
    Pedroso, M., & Ogasawara, E. (2020). An analysis of malaria in the Brazilian Legal
    Amazon using divergent association rules. Journal of Biomedical Informatics, 108, 103512.

    Porting
    -------
    Canonical implementation; the R ``pat_dara.R`` is a backport of this.
    """
    data = adjust_dataframe(data)
    if rhs_attr is None or rhs_attr not in data.columns:
        raise ValueError("dara: 'rhs_attr' must be a column of data.")
    if rhs_value is not None:
        data = data[data[rhs_attr] == rhs_value]

    parsed = [parse_items(a) for a in rules["antecedents"]] if len(rules) else []
    weights = None
    if rule_weight is not None:
        if rule_weight not in rules.columns:
            raise ValueError(f"dara: 'rule_weight' column '{rule_weight}' not found in rules.")
        weights = pd.to_numeric(rules[rule_weight], errors="coerce").fillna(0.0).to_numpy()

    if attributes is None:
        rule_attrs = set().union(*parsed) if parsed else set()
        attributes = [c for c in data.columns if c in rule_attrs and c != rhs_attr]
    else:
        attributes = [a for a in attributes if a in data.columns and a != rhs_attr]

    results: dict = {}
    for attribute in attributes:
        data_freq = data[attribute].value_counts()
        if data_freq.empty:
            continue
        rule_freq = _rule_value_counts(parsed, attribute, weights)
        if not rule_freq:
            continue
        values = [str(v) for v in data_freq.index]
        ds = data_freq.to_numpy(dtype=float)
        rl = np.array([float(rule_freq.get(v, 0.0)) for v in values])
        counter = _dara_counter(ds, rl)
        table = pd.DataFrame({"value": values, "dataset": ds, "rules": rl, "c": counter})
        results[attribute] = table.sort_values(["c", "value"]).reset_index(drop=True)
    return results


def dara_score(ranking: Mapping, antecedent, how: str = "sum", use_abs: bool = False) -> float:
    """Aggregate the DARA divergence counters of one rule's antecedent values.

    ``antecedent`` is an iterable of ``attribute=value`` item labels; each is looked up in
    ``ranking`` and the counters are combined by ``how`` (``"sum"``/``"mean"``/``"max"``).
    """
    vals = []
    for attribute, value in parse_items(antecedent).items():
        table = ranking.get(attribute)
        if table is None:
            continue
        hit = table.loc[table["value"] == value, "c"]
        if len(hit):
            vals.append(float(hit.iloc[0]))
    if not vals:
        return 0.0
    arr = np.abs(vals) if use_abs else np.asarray(vals, dtype=float)
    if how == "sum":
        return float(arr.sum())
    if how == "mean":
        return float(arr.mean())
    if how == "max":
        return float(arr.max())
    raise ValueError("dara_score: 'how' must be 'sum', 'mean', or 'max'.")

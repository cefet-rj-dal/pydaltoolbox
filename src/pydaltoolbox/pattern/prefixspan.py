"""Sequential pattern mining with PrefixSpan (``pat_cspade.R`` analog).

PrefixSpan mines frequent sequential patterns -- ordered lists of itemsets (events) that
occur, in order, in a fraction of the input sequences at or above a support threshold.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import (
    PatternMiner,
    metric_mean_support,
    metric_pattern_count,
    metric_retained_ratio,
)
from .threshold import SupportThreshold

__all__ = ["PatPrefixSpan"]

# A normalized sequence is a list of (time, items) events, items a sorted tuple, time sorted.
Event = tuple  # (time, tuple_of_items)


def _normalize(data) -> list:
    """Normalize the input into a list of sequences of ``(time, sorted_items)`` events.

    Accepts a long :class:`pandas.DataFrame` (``sequenceID``/``eventID``/``item`` columns,
    mirroring ``arulesSequences::read_baskets``) or a list of sequences whose events are
    item collections (event position used as time) or ``(time, items)`` pairs.
    """
    if isinstance(data, pd.DataFrame):
        cols = {c.lower(): c for c in data.columns}
        sid = cols.get("sequenceid")
        eid = cols.get("eventid")
        item = cols.get("item", cols.get("items"))
        if sid is None or eid is None or item is None:
            raise ValueError(
                "PatPrefixSpan: DataFrame input needs 'sequenceID', 'eventID' and 'item' columns."
            )
        sequences = []
        for _, grp in data.groupby(sid, sort=True):
            events = []
            for ev, evgrp in grp.groupby(eid, sort=True):
                items = tuple(sorted({str(v) for v in evgrp[item]}))
                events.append((ev, items))
            sequences.append(events)
        return sequences

    sequences = []
    for seq in data:
        events = []
        for t, event in enumerate(seq):
            if (
                isinstance(event, tuple)
                and len(event) == 2
                and not isinstance(event[1], str)
                and isinstance(event[1], Iterable)
            ):
                time, items = event
            else:
                time, items = t, event
            events.append((time, tuple(sorted({str(v) for v in items}))))
        events.sort(key=lambda e: e[0])
        sequences.append(events)
    return sequences


def _sequence_label(pattern) -> str:
    """Format a pattern (list of item-tuples) like ``<{A,B},{C}>``."""
    return "<" + ",".join("{" + ",".join(elem) + "}" for elem in pattern) + ">"


class PatPrefixSpan(PatternMiner):
    """PrefixSpan sequential pattern mining.

    PrefixSpan (Prefix-projected Sequential pattern mining) discovers frequent sequential
    patterns -- ordered lists of itemsets -- by growing patterns one prefix at a time and
    recursively mining the *projected database* (the suffixes that follow each occurrence
    of the current prefix). Each discovered pattern is grown by either an *i-extension*
    (adding an item to its last itemset) or an *s-extension* (appending a new itemset),
    and the support of a pattern is the fraction of input sequences in which it occurs in
    order. ``maxlen`` bounds the number of itemsets, ``maxsize`` the items per itemset, and
    ``mingap``/``maxgap`` constrain the time difference between consecutive matched events.

    Sequences are given as a list of sequences (each a list of itemsets, or of
    ``(time, itemset)`` events) or as a long DataFrame with ``sequenceID``/``eventID``/
    ``item`` columns. When ``support`` is ``0`` it is estimated during ``fit`` from the data
    using ``support_strategy``.

    Porting:
        Canonical implementation (PrefixSpan, Pei et al., 2001); the R
        ``pat_prefix_span.R`` is a backport of this. Fidelity: adapted.
        - The itemset pseudo-projection tracks, per sequence, the set of elements the
          prefix's last element can map to; a naive first-occurrence projection would
          undercount itemset patterns.
        - ``mingap``/``maxgap`` are applied on the event times during projection.
        - Validated identical to the classic Zaki example (18/11/17/12 frequent patterns
          for support=0.5 / maxgap=5 / maxgap=10 / mingap=10).

    References:
        Pei, J., Han, J., Mortazavi-Asl, B., Wang, J., Pinto, H., Chen, Q., Dayal, U., &
        Hsu, M. (2001). PrefixSpan: Mining Sequential Patterns Efficiently by
        Prefix-Projected Pattern Growth. ICDE, 215-224.
    """

    def __init__(
        self,
        support: float = 0.0,
        minlen: int = 1,
        maxlen: int | None = None,
        maxsize: int | None = None,
        mingap: float | None = None,
        maxgap: float | None = None,
        quality_filter: dict | None = None,
        support_strategy: SupportThreshold | None = None,
    ) -> None:
        super().__init__()
        if support < 0:
            raise ValueError("PatPrefixSpan: support must be >= 0 (0 = estimate in fit).")
        self.support = support
        self.minlen = minlen
        self.maxlen = maxlen
        self.maxsize = maxsize
        self.mingap = mingap
        self.maxgap = maxgap
        self.quality_filter = quality_filter
        self.support_strategy = support_strategy or SupportThreshold("min_count", min_count=2)
        self.pattern_kind = "sequences"
        self._sequences: list | None = None
        self._support_resolved: float | None = None
        self.eval_metrics = [
            metric_pattern_count,
            metric_mean_support,
            _metric_mean_length,
            metric_retained_ratio,
        ]

    # -- contract ---------------------------------------------------------------
    def fit(self, data, **kwargs) -> PatPrefixSpan:
        sequences = _normalize(data)
        self._sequences = sequences
        self.schema = sorted({it for seq in sequences for _, items in seq for it in items})
        self.data_class = type(data).__name__
        self.fitted = True
        if self.support and self.support > 0:
            self._support_resolved = self.support
        else:
            self._support_resolved = self.support_strategy.resolve(self._sequence_onehot(sequences))
        return self

    def action(self, data=None, **kwargs):
        """For pattern miners, ``action`` dispatches to :meth:`discover`."""
        return self.discover(data, **kwargs)

    def discover(self, data=None, **kwargs) -> pd.DataFrame:
        self._require_fitted()
        sequences = self._sequences if data is None else _normalize(data)
        n = len(sequences)
        min_count = max(1, int(np.ceil(self._support_resolved * n - 1e-9)))

        patterns = _prefixspan(
            sequences, min_count,
            maxlen=self.maxlen or 10**9,
            maxsize=self.maxsize or 10**9,
            mingap=self.mingap,
            maxgap=self.maxgap,
        )
        rows = [
            {
                "sequence": _sequence_label(pat),
                "elements": [frozenset(e) for e in pat],
                "support": count / n,
                "length": len(pat),
            }
            for pat, count in patterns
            if len(pat) >= self.minlen
        ]
        out = pd.DataFrame(rows, columns=["sequence", "elements", "support", "length"])
        total = len(out)
        out = self._apply_quality_filter(out)
        out = out.sort_values(["support", "sequence"], ascending=[False, True]).reset_index(
            drop=True
        )
        out.attrs["filtered_from"] = total
        return out

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _sequence_onehot(sequences) -> pd.DataFrame:
        """One row per sequence, one boolean column per item present anywhere in it."""
        items = sorted({it for seq in sequences for _, ev in seq for it in ev})
        rows = [{it: any(it in ev for _, ev in seq) for it in items} for seq in sequences]
        return pd.DataFrame(rows, columns=items)


def _metric_mean_length(patterns: pd.DataFrame, obj=None) -> dict:
    value = float(patterns["length"].mean()) if len(patterns) else float("nan")
    return {"metric": "mean_length", "value": value, "type": "intrinsic"}


def _prefixspan(sequences, min_count, maxlen, maxsize, mingap, maxgap):
    """PrefixSpan returning ``(pattern, support_count)`` for every frequent pattern.

    Projection tracks, per sequence, the *set* of elements onto which the prefix's last
    element can be mapped (so a later element that fully contains the extended itemset is
    not missed). An i-extension adds an item to the last element; an s-extension appends a
    new element subject to the ``mingap``/``maxgap`` time constraints.
    """
    results: list = []

    def recurse(prefix, proj, is_empty):
        last_item = prefix[-1][-1] if not is_empty else None
        n_elem = len(prefix)
        last_size = len(prefix[-1]) if not is_empty else 0

        i_cand: dict = {}  # item -> {seq_idx: set(element_idx)}
        s_cand: dict = {}
        for si, eset in proj.items():
            seq = sequences[si]
            if not is_empty and last_size < maxsize:
                for e in eset:
                    for x in seq[e][1]:
                        if x > last_item:
                            i_cand.setdefault(x, {}).setdefault(si, set()).add(e)
            if n_elem < maxlen:
                for e_prev in eset:
                    t_prev = None if e_prev < 0 else seq[e_prev][0]
                    for e2 in range(e_prev + 1, len(seq)):
                        if t_prev is not None:
                            gap = seq[e2][0] - t_prev
                            if maxgap is not None and gap > maxgap:
                                break
                            if mingap is not None and gap < mingap:
                                continue
                        for x in seq[e2][1]:
                            s_cand.setdefault(x, {}).setdefault(si, set()).add(e2)

        for x in sorted(i_cand):
            seqs = i_cand[x]
            if len(seqs) >= min_count:
                newprefix = prefix[:-1] + (prefix[-1] + (x,),)
                results.append((newprefix, len(seqs)))
                recurse(newprefix, seqs, False)

        for x in sorted(s_cand):
            seqs = s_cand[x]
            if len(seqs) >= min_count:
                newprefix = prefix + ((x,),)
                results.append((newprefix, len(seqs)))
                recurse(newprefix, seqs, False)

    initial = {si: {-1} for si in range(len(sequences)) if sequences[si]}
    recurse((), initial, True)
    return results

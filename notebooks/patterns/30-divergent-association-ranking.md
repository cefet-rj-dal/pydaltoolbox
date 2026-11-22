---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Divergent Association Rules (DARA)

DARA ranks the values of each explanatory attribute by how much their prominence in a set
of association rules **diverges** from their frequency in the data. It assumes attribute-value
independence: under that hypothesis, a value that is more frequent in the data should appear
in *more* rules than a less frequent value. When that expectation is violated — a frequent
value appears in few rules, or a rare value appears in many — the value is *divergent*, and
rules built on it carry non-trivial structure worth inspecting.

For each attribute, DARA compares two orderings of its values (by data frequency and by rule
count) and accumulates a divergence counter `c` over every pair of values: a value more
frequent in the data but appearing in fewer rules moves negative, and the reverse moves
positive.

**Reference:** Baroni, L., Salles, R., Salles, S., Guedes, G., Porto, F., Bezerra, E.,
Barcellos, C., Pedroso, M., & Ogasawara, E. (2020). *An analysis of malaria in the Brazilian
Legal Amazon using divergent association rules.* Journal of Biomedical Informatics, 108, 103512.

> Pattern mining needs the optional `mlxtend` dependency: `pip install pydaltoolbox[patterns]`.

## The divergence counter

Take an attribute `col` whose value `A` is frequent in the data (5 rows) yet appears in only
one rule, while the rare value `B` (1 row) appears in three rules. `A` is less prominent in
the rules than its data frequency predicts, so it scores negative; `B` scores positive.

```python
import pandas as pd

import pydaltoolbox as dal

data = pd.DataFrame({"col": list("AAAAAB"), "y": ["s"] * 6})
rules = pd.DataFrame(
    {
        "antecedents": [frozenset(["col=A"])] + [frozenset(["col=B"])] * 3,
        "consequents": [frozenset(["y=s"])] * 4,
    }
)

dal.dara(data, rules, rhs_attr="y", rhs_value="s")["col"]
```

## Mining and filtering rules end to end

Transactions are a one-hot `DataFrame` whose items are `attribute=value` labels (here via
`pandas.get_dummies` with `prefix_sep="="`). `PatApriori` mines rules for a fixed right-hand
side, and a `RuleFilterDara` scores each rule by the DARA divergence of its antecedent values
and keeps the divergent ones.

In the data below, `region=A` is the most common region among positive outcomes yet predicts
them only half the time, so it generates no confident rule; `region=C` is rarer but always
positive. `C` is therefore divergent — prominent in the rules beyond what its frequency
predicts — and DARA surfaces the rule built on it.

```python
raw = pd.DataFrame(
    {
        "region":  ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
        "season":  ["wet", "dry", "wet", "dry", "wet", "dry", "wet", "dry", "wet", "dry", "wet", "dry"],
        "outcome": ["pos", "pos", "pos", "neg", "neg", "neg", "pos", "pos", "neg", "neg", "pos", "pos"],
    }
)
transactions = pd.get_dummies(raw, prefix_sep="=")

dara_filter = dal.RuleFilterDara(raw, rhs_attr="outcome", rhs_value="pos", min_score=1)
miner = dal.PatApriori(
    target="rules", supp=0.1, conf=0.6, minlen=1, maxlen=2,
    rhs=["outcome=pos"], rule_filter=dara_filter,
)
miner.fit(transactions)
rules = miner.discover(transactions)
rules[["antecedents", "consequents", "support", "confidence", "lift", "dara_score"]]
```

The per-attribute divergence ranking used for scoring is kept alongside the result — here
`region=A` scores negative (frequent in the data, absent from the rules) and `region=C`
scores positive:

```python
rules.attrs["dara_ranking"]["region"]
```

## Other rule filters: none and interest measures

Besides DARA, rules can be kept unchanged (`RuleFilterNone`) or filtered by classical
interest measures (`RuleFilterInterest`: minimum `lift`, Kulczynski `kulc`, support,
confidence and count, and maximum imbalance ratio `ir`).

```python
transactions = pd.get_dummies(raw, prefix_sep="=")
interest = dal.RuleFilterInterest(lift_min=1.0, kulc_min=0.5)
miner = dal.PatApriori(
    target="rules", supp=0.1, conf=0.6, minlen=1, maxlen=2,
    rhs=["outcome=pos"], rule_filter=interest,
)
miner.fit(transactions)
miner.discover(transactions)[["antecedents", "consequents", "support", "confidence", "lift", "kulc", "ir"]]
```

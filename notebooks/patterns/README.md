# Pattern Mining Examples

This section introduces pattern mining as a family of descriptive learners that discover structure directly from co-occurrence data. The numbering now leaves semantic gaps because association rules, itemsets, and sequential patterns are related but conceptually distinct mining tasks.

The didactic question here is different from classification and regression: how do we configure the search space, constrain the discovered patterns, and summarize their quality in a reproducible way?

When `supp`, `conf`, or `support` are set to `0`, the miner estimates them during `fit()` with threshold strategies such as `SupportThreshold` and `ConfidenceThreshold`. These strategies configure the search space; the filters below are post-mining rule filters.

## Association Rules

Start here if you want directional patterns with left-hand side and right-hand side interpretation.

- [01-apriori-rules.md](01-apriori-rules.md) - `PatApriori`: directional rules with a fixed consequent (`rhs`) and explicit filtering by confidence and lift.

## Filtering and Constraints

Before sequence mining, it helps to separate two kinds of intelligence: directional rules and filtered co-occurrence structures.

- [10-apriori-itemsets.md](10-apriori-itemsets.md) - `PatApriori` (frequent itemsets): itemset mining with `include`, `exclude`, and post-mining support filtering.

## Sequential Patterns

Finish with sequence mining, where order matters and the interpretation shifts from co-occurrence to recurring event structure.

- [20-prefixspan-sequences.md](20-prefixspan-sequences.md) - `PatPrefixSpan`: mines sequential patterns and evaluates them through support and sequence summaries.

## Interesting Rule Filters

Use interesting-rule filters after a miner has produced candidate association rules. The default `RuleFilterNone` keeps all candidate rules; classical filters use measures such as lift, Kulczynski, and imbalance ratio; DARA filters rules by divergent antecedent values.

- [30-divergent-association-ranking.md](30-divergent-association-ranking.md) - `pat_rule_filter_none`, `RuleFilterInterest`, `RuleFilterDara`, and rule filtering: post-mining filtering of interesting association rules.

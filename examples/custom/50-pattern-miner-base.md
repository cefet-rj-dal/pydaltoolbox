# Pattern Miner Base (PatternMiner)

`PatternMiner` is the common contract behind the pattern-discovery learners. Unlike the
algorithm-specific examples, the goal here is not to discover patterns, but to inspect the
workflow expectations shared by `PatApriori` and `PatPrefixSpan`.

## Inspect the base contract


```python
import pydaltoolbox as dal

miner = dal.PatternMiner()
{
    "type": type(miner).__name__,
    "pattern_kind": miner.pattern_kind,
    "has_fit": hasattr(miner, "fit"),
    "has_discover": hasattr(miner, "discover"),
    "has_evaluate": hasattr(miner, "evaluate"),
}
```




    {'type': 'PatternMiner',
     'pattern_kind': 'patterns',
     'has_fit': True,
     'has_discover': True,
     'has_evaluate': True}



## What to observe
- Pattern miners revolve around `fit`, `discover`, and `evaluate`.
- Algorithm-specific constructors mostly specialize engine configuration, filtering, and
  quality summaries on top of this base object.

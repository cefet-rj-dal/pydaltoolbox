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

# Autoencoder Base Encoder (AutoencBaseE)

`AutoencBaseE` is an **identity base class**: it defines the contract (an input size and a
latent size, plus `fit`/`transform`) but does no learning by itself. Real implementations --
a deep-learning autoencoder in a companion package, or detectors in `pyharbinger` -- subclass
it and override the two methods. Keeping the base here, with no deep-learning dependency, is
what lets downstream packages extend a stable interface.

## The base is identity
```python
import pydaltoolbox as dal

iris = dal.datasets.iris().drop(columns="Species")

base = dal.AutoencBaseE(input_size=4, encoding_size=2).fit(iris)
base.transform(iris).head()
```

## A concrete encoder overrides `fit` and `transform`
Because it respects the same contract, a PCA-based encoder drops straight into any workflow.
```python
import pandas as pd
from sklearn.decomposition import PCA


class PcaEncoder(dal.AutoencBaseE):
    def fit(self, data, **kwargs):
        self.model = PCA(n_components=self.encoding_size).fit(data.to_numpy(float))
        return self

    def transform(self, data, **kwargs):
        codes = self.model.transform(data.to_numpy(float))
        return pd.DataFrame(codes, columns=["z1", "z2"], index=data.index)


PcaEncoder(input_size=4, encoding_size=2).fit(iris).transform(iris).head()
```

## What to observe
- The base already participates in the DAL transformation protocol.
- The identity behavior is deliberate: this file teaches the extension contract, not a
  finished neural implementation.

## References
- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with
  Neural Networks.* Science.

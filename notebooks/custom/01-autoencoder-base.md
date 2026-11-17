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

# Autoencoder Base Classes (AutoencBaseE / AutoencBaseEd)

`AutoencBaseE` (encoder) and `AutoencBaseEd` (encoder + decoder) are **identity base
classes**: they define the contract (an input size and a latent size, plus
`fit`/`transform`) but do no learning by themselves. The real implementations -- a
deep-learning backed autoencoder in a companion package, or the detectors in
`pyharbinger` -- subclass them and override the two methods. Keeping the base here, with
no deep-learning dependency, is what lets downstream packages extend a stable interface.

**Reference:** Hinton, G. E., Salakhutdinov, R. R. (2006). *Reducing the Dimensionality
of Data with Neural Networks.* Science.

```python
import pydaltoolbox as dal

iris = dal.datasets.iris().drop(columns="Species")

# the base is identity: transform returns the input unchanged
base = dal.AutoencBaseE(input_size=4, encoding_size=2).fit(iris)
base.transform(iris).head()
```

A concrete encoder just overrides `fit` and `transform` -- here a PCA-based encoder.
Because it respects the same contract, it drops straight into any workflow.

```python
from sklearn.decomposition import PCA

import pandas as pd


class PcaEncoder(dal.AutoencBaseE):
    def fit(self, data, **kwargs):
        self.model = PCA(n_components=self.encoding_size).fit(data.to_numpy(float))
        return self

    def transform(self, data, **kwargs):
        codes = self.model.transform(data.to_numpy(float))
        return pd.DataFrame(codes, columns=["z1", "z2"], index=data.index)


encoder = PcaEncoder(input_size=4, encoding_size=2).fit(iris)
encoder.transform(iris).head()
```

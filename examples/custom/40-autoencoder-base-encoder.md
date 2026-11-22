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




       sepal_length  sepal_width  petal_length  petal_width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    2           4.7          3.2           1.3          0.2
    3           4.6          3.1           1.5          0.2
    4           5.0          3.6           1.4          0.2



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




             z1        z2
    0 -2.684126  0.319397
    1 -2.714142 -0.177001
    2 -2.888991 -0.144949
    3 -2.745343 -0.318299
    4 -2.728717  0.326755



## What to observe
- The base already participates in the DAL transformation protocol.
- The identity behavior is deliberate: this file teaches the extension contract, not a
  finished neural implementation.

## References
- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with
  Neural Networks.* Science.

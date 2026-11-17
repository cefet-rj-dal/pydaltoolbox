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




       sepal_length  sepal_width  petal_length  petal_width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    2           4.7          3.2           1.3          0.2
    3           4.6          3.1           1.5          0.2
    4           5.0          3.6           1.4          0.2



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




             z1        z2
    0 -2.684126  0.319397
    1 -2.714142 -0.177001
    2 -2.888991 -0.144949
    3 -2.745343 -0.318299
    4 -2.728717  0.326755



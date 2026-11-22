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

# Autoencoder Base Encoder-Decoder (AutoencBaseEd)

`AutoencBaseEd` extends the encoder contract with a decoder: a concrete subclass encodes the
input to a latent space and reconstructs it back to the original dimensionality. As with
`AutoencBaseE`, the base is identity so downstream packages can override `fit`/`transform`
with a real neural backend.

## The base is identity
```python
import pydaltoolbox as dal

iris = dal.datasets.iris().drop(columns="Species")

base = dal.AutoencBaseEd(input_size=4, encoding_size=2).fit(iris)
base.transform(iris).head()
```

## A concrete encoder-decoder overrides `fit` and `transform`
Here a PCA-based encoder-decoder: it projects to the latent space and reconstructs the input,
so the output has the original number of columns.
```python
import pandas as pd
from sklearn.decomposition import PCA


class PcaEncoderDecoder(dal.AutoencBaseEd):
    def fit(self, data, **kwargs):
        self.model = PCA(n_components=self.encoding_size).fit(data.to_numpy(float))
        self.columns = list(data.columns)
        return self

    def transform(self, data, **kwargs):
        codes = self.model.transform(data.to_numpy(float))
        recon = self.model.inverse_transform(codes)
        return pd.DataFrame(recon, columns=self.columns, index=data.index)


recon = PcaEncoderDecoder(input_size=4, encoding_size=2).fit(iris).transform(iris)
recon.head()
```

## Reconstruction error
```python
((iris - recon) ** 2).mean().mean()
```

## References
- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with
  Neural Networks.* Science.

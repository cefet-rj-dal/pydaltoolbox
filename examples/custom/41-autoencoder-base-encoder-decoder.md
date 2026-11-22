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




       sepal_length  sepal_width  petal_length  petal_width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    2           4.7          3.2           1.3          0.2
    3           4.6          3.1           1.5          0.2
    4           5.0          3.6           1.4          0.2



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




       sepal_length  sepal_width  petal_length  petal_width
    0      5.083039     3.517414      1.403214     0.213532
    1      4.746262     3.157500      1.463562     0.240246
    2      4.704119     3.195682      1.308217     0.175180
    3      4.642212     3.056967      1.461330     0.239732
    4      5.071755     3.526555      1.363738     0.197000



## Reconstruction error


```python
((iris - recon) ** 2).mean().mean()
```




    np.float64(0.025341073932398255)



## References
- Hinton, G. E., and Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with
  Neural Networks.* Science.

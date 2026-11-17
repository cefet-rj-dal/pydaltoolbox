"""Unit tests for the autoencoder base classes."""
from pydaltoolbox import AutoencBaseE, AutoencBaseEd, datasets


def test_bases_store_sizes():
    enc = AutoencBaseE(input_size=4, encoding_size=2)
    assert enc.input_size == 4 and enc.encoding_size == 2
    ed = AutoencBaseEd(input_size=4, encoding_size=2)
    assert ed.input_size == 4 and ed.encoding_size == 2


def test_fit_is_chainable_and_transform_is_identity():
    iris = datasets.iris().drop(columns="Species")
    for cls in (AutoencBaseE, AutoencBaseEd):
        obj = cls(input_size=4, encoding_size=2)
        assert obj.fit(iris) is obj
        assert obj.transform(iris).equals(iris)  # identity by default


def test_subclass_can_override_to_encode():
    """A subclass overrides fit/transform; the base contract makes it drop-in."""
    from sklearn.decomposition import PCA

    class PcaEncoder(AutoencBaseE):
        def fit(self, data, **kwargs):
            self.model = PCA(n_components=self.encoding_size).fit(data.to_numpy(float))
            return self

        def transform(self, data, **kwargs):
            return self.model.transform(data.to_numpy(float))

    iris = datasets.iris().drop(columns="Species")
    enc = PcaEncoder(input_size=4, encoding_size=2).fit(iris)
    out = enc.transform(iris)
    assert out.shape == (len(iris), 2)

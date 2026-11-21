"""Autoencoder base classes.

Python counterparts of ``autoenc_base_e.R`` (encoder) and ``autoenc_base_ed.R``
(encoder + decoder). As in the R package, these are **identity** base classes: they do
not train or transform by themselves. Concrete implementations (e.g. a deep-learning
backed autoencoder in a companion package, or the detectors in pyharbinger) subclass
them and override ``fit``/``transform``. They exist here so that downstream code has a
stable base to extend, without pulling a deep-learning dependency into pydaltoolbox.
"""
from __future__ import annotations

from .base import DalTransform

__all__ = ["AutoencBaseE", "AutoencBaseEd"]


class AutoencBaseE(DalTransform):
    """Base class for encoder-only autoencoders (identity by default).

    An autoencoder is a neural network trained to reconstruct its input through a
    lower-dimensional bottleneck, so the bottleneck learns a compressed (non-linear)
    encoding of the data -- a non-linear generalization of PCA. This encoder-only base
    exposes just the input-to-code mapping; by default it is the identity, so a subclass
    (e.g. a deep-learning-backed autoencoder) supplies the real ``fit``/``transform``.

    Parameters
    ----------
    input_size:
        Dimensionality of the input vector.
    encoding_size:
        Dimensionality of the latent (encoded) representation.

    References:
        Hinton, G. E., and Salakhutdinov, R. R. (2006). Reducing the Dimensionality of
        Data with Neural Networks. Science, 313(5786):504-507.
    """

    def __init__(self, input_size: int, encoding_size: int) -> None:
        self.input_size = input_size
        self.encoding_size = encoding_size

    def fit(self, data, **kwargs) -> AutoencBaseE:
        return self

    def transform(self, data, **kwargs):
        return data


class AutoencBaseEd(DalTransform):
    """Base class for encoder + decoder autoencoders (identity by default).

    Like :class:`AutoencBaseE`, but exposes the full encode-then-decode round trip: the
    input is mapped to a latent code and back to the original space. The reconstruction
    (and its error) is useful for denoising, anomaly detection and generative modeling.
    By default it is the identity, so a subclass supplies the real ``fit``/``transform``.

    Parameters
    ----------
    input_size:
        Dimensionality of the input vector.
    encoding_size:
        Dimensionality of the latent (encoded) representation.

    References:
        Hinton, G. E., and Salakhutdinov, R. R. (2006). Reducing the Dimensionality of
        Data with Neural Networks. Science, 313(5786):504-507.
    """

    def __init__(self, input_size: int, encoding_size: int) -> None:
        self.input_size = input_size
        self.encoding_size = encoding_size

    def fit(self, data, **kwargs) -> AutoencBaseEd:
        return self

    def transform(self, data, **kwargs):
        return data

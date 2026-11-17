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

    Parameters
    ----------
    input_size:
        Dimensionality of the input vector.
    encoding_size:
        Dimensionality of the latent (encoded) representation.
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

    Parameters
    ----------
    input_size:
        Dimensionality of the input vector.
    encoding_size:
        Dimensionality of the latent (encoded) representation.
    """

    def __init__(self, input_size: int, encoding_size: int) -> None:
        self.input_size = input_size
        self.encoding_size = encoding_size

    def fit(self, data, **kwargs) -> AutoencBaseEd:
        return self

    def transform(self, data, **kwargs):
        return data

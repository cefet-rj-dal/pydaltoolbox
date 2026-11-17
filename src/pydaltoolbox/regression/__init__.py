"""Regression family."""
from .base import Regression, SklearnRegression
from .dtree import RegDtree
from .ensemble import RegRf
from .knn import RegKnn
from .lm import RegLm
from .neural import RegMlp
from .svm import RegSvm

__all__ = [
    "RegDtree",
    "RegKnn",
    "RegLm",
    "RegMlp",
    "RegRf",
    "RegSvm",
    "Regression",
    "SklearnRegression",
]

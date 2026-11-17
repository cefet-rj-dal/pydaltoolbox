"""Classification family."""
from .base import Classification, SklearnClassification
from .baseline import ClaMajority
from .dtree import ClaDtree, ClaRpart
from .ensemble import ClaBagging, ClaBoosting, ClaRf
from .knn import ClaKnn
from .linear import ClaGlm, ClaGlmnet, ClaMultinom
from .naive_bayes import ClaNb
from .neural import ClaMlp
from .svm import ClaSvm

__all__ = [
    "ClaBagging",
    "ClaBoosting",
    "ClaDtree",
    "ClaGlm",
    "ClaGlmnet",
    "ClaKnn",
    "ClaMajority",
    "ClaMlp",
    "ClaMultinom",
    "ClaNb",
    "ClaRf",
    "ClaRpart",
    "ClaSvm",
    "Classification",
    "SklearnClassification",
]

"""pydaltoolbox -- Leveraging Experiment Lines for Data Analytics.

Python port of the R ``daltoolbox`` framework. The public API exposes the
Experiment Line base classes plus concrete components, all following the uniform
``fit`` / ``action`` / ``evaluate`` contract.
"""
from __future__ import annotations

from . import datasets
from .adjust import (
    adjust_class_label,
    adjust_dataframe,
    adjust_factor,
    adjust_matrix,
)
from .base import DalBase, DalLearner, DalTransform, Predictor
from .classification import (
    ClaBagging,
    ClaBoosting,
    ClaDtree,
    ClaGlm,
    ClaGlmnet,
    ClaKnn,
    ClaMajority,
    ClaMlp,
    ClaMultinom,
    ClaNb,
    ClaRf,
    ClaRpart,
    Classification,
    ClaSvm,
    SklearnClassification,
)
from .regression import (
    RegDtree,
    RegKnn,
    RegLm,
    RegMlp,
    Regression,
    RegRf,
    RegSvm,
    SklearnRegression,
)
from .sampling import DataSample, SampleRandom, SampleStratified, train_test_from_folds
from .transform import (
    BalOversampling,
    BalSubsampling,
    CategMapping,
    DtPca,
    FeatureSelectionCorr,
    FeatureSelectionInfoGain,
    FeatureSelectionLasso,
    FeatureSelectionStepwise,
    FitCurvatureMax,
    FitCurvatureMin,
    ImputationSimple,
    MinMax,
    NaRemoval,
    OutliersBoxplot,
    OutliersGaussian,
    SampleBalance,
    ZScore,
)

__version__ = "0.1.0"

__all__ = [
    "BalOversampling",
    "BalSubsampling",
    "CategMapping",
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
    "DalBase",
    "DalLearner",
    "DalTransform",
    "DataSample",
    "DtPca",
    "FeatureSelectionCorr",
    "FeatureSelectionInfoGain",
    "FeatureSelectionLasso",
    "FeatureSelectionStepwise",
    "FitCurvatureMax",
    "FitCurvatureMin",
    "ImputationSimple",
    "MinMax",
    "NaRemoval",
    "OutliersBoxplot",
    "OutliersGaussian",
    "Predictor",
    "RegDtree",
    "RegKnn",
    "RegLm",
    "RegMlp",
    "RegRf",
    "RegSvm",
    "Regression",
    "SampleBalance",
    "SampleRandom",
    "SampleStratified",
    "SklearnClassification",
    "SklearnRegression",
    "ZScore",
    "adjust_class_label",
    "adjust_dataframe",
    "adjust_factor",
    "adjust_matrix",
    "datasets",
    "train_test_from_folds",
]

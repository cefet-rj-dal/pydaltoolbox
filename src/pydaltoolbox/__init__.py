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
from .autoenc import AutoencBaseE, AutoencBaseEd
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
    ClaXgboost,
    SklearnClassification,
)
from .clustering import (
    ClusterCmeans,
    ClusterDbscan,
    Clusterer,
    ClusterGmm,
    ClusterHclust,
    ClusterKmeans,
    ClusterPam,
)
from .pattern import (
    ConfidenceThreshold,
    PatApriori,
    PatPrefixSpan,
    PatternMiner,
    PatUtils,
    RuleFilter,
    RuleFilterDara,
    RuleFilterInterest,
    RuleFilterNone,
    SupportThreshold,
    dara,
    dara_score,
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
from .sampling import (
    DataSample,
    SampleGroups,
    SampleRandom,
    SampleSimple,
    SampleStratified,
    train_test_from_folds,
)
from .transform import (
    Aggregation,
    BalOversampling,
    BalSubsampling,
    CategMapping,
    DtPca,
    FeatureGeneration,
    FeatureSelectionCorr,
    FeatureSelectionFss,
    FeatureSelectionInfoGain,
    FeatureSelectionLasso,
    FeatureSelectionRelief,
    FeatureSelectionStepwise,
    FitCurvatureMax,
    FitCurvatureMin,
    HierarchyCut,
    ImputationPredictive,
    ImputationSimple,
    ImputationTree,
    MinMax,
    NaRemoval,
    OutliersBoxplot,
    OutliersGaussian,
    SampleBalance,
    Smoothing,
    SmoothingCluster,
    SmoothingFreq,
    SmoothingInter,
    SmoothingQuantization,
    ZScore,
)
from .tune import ClaTune, CluTune, DalTune, RegTune

__version__ = "0.15.0"

# Visualization helpers live in the optional ``[viz]`` extra (matplotlib). They
# are resolved lazily (PEP 562) so importing pydaltoolbox never requires
# matplotlib; a clear ImportError surfaces only when a ``plot_*`` is first used.
_VIZ_EXPORTS = (
    "plot_bar",
    "plot_boxplot",
    "plot_boxplot_class",
    "plot_correlation",
    "plot_dendrogram",
    "plot_density",
    "plot_density_class",
    "plot_groupedbar",
    "plot_hist",
    "plot_lollipop",
    "plot_pair",
    "plot_parallel",
    "plot_pieplot",
    "plot_pixel",
    "plot_points",
    "plot_radar",
    "plot_scatter",
    "plot_series",
    "plot_stackedbar",
    "plot_ts",
    "plot_ts_pred",
)


def __getattr__(name: str):
    if name in _VIZ_EXPORTS:
        try:
            from . import viz
        except ImportError as exc:  # pragma: no cover - exercised without matplotlib
            raise ImportError(
                f"{name!r} requires the optional 'viz' extra. "
                "Install it with: pip install pydaltoolbox[viz]"
            ) from exc
        return getattr(viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_VIZ_EXPORTS))

__all__ = [
    "Aggregation",
    "AutoencBaseE",
    "AutoencBaseEd",
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
    "ClaTune",
    "ClaXgboost",
    "Classification",
    "CluTune",
    "ClusterCmeans",
    "ClusterDbscan",
    "ClusterGmm",
    "ClusterHclust",
    "ClusterKmeans",
    "ClusterPam",
    "Clusterer",
    "ConfidenceThreshold",
    "DalBase",
    "DalLearner",
    "DalTransform",
    "DalTune",
    "DataSample",
    "DtPca",
    "FeatureGeneration",
    "FeatureSelectionCorr",
    "FeatureSelectionFss",
    "FeatureSelectionInfoGain",
    "FeatureSelectionLasso",
    "FeatureSelectionRelief",
    "FeatureSelectionStepwise",
    "FitCurvatureMax",
    "FitCurvatureMin",
    "HierarchyCut",
    "ImputationPredictive",
    "ImputationSimple",
    "ImputationTree",
    "MinMax",
    "NaRemoval",
    "OutliersBoxplot",
    "OutliersGaussian",
    "PatApriori",
    "PatPrefixSpan",
    "PatUtils",
    "PatternMiner",
    "Predictor",
    "RegDtree",
    "RegKnn",
    "RegLm",
    "RegMlp",
    "RegRf",
    "RegSvm",
    "RegTune",
    "Regression",
    "RuleFilter",
    "RuleFilterDara",
    "RuleFilterInterest",
    "RuleFilterNone",
    "SampleBalance",
    "SampleGroups",
    "SampleRandom",
    "SampleSimple",
    "SampleStratified",
    "SklearnClassification",
    "SklearnRegression",
    "Smoothing",
    "SmoothingCluster",
    "SmoothingFreq",
    "SmoothingInter",
    "SmoothingQuantization",
    "SupportThreshold",
    "ZScore",
    "adjust_class_label",
    "adjust_dataframe",
    "adjust_factor",
    "adjust_matrix",
    "dara",
    "dara_score",
    "datasets",
    "plot_bar",
    "plot_boxplot",
    "plot_boxplot_class",
    "plot_correlation",
    "plot_dendrogram",
    "plot_density",
    "plot_density_class",
    "plot_groupedbar",
    "plot_hist",
    "plot_lollipop",
    "plot_pair",
    "plot_parallel",
    "plot_pieplot",
    "plot_pixel",
    "plot_points",
    "plot_radar",
    "plot_scatter",
    "plot_series",
    "plot_stackedbar",
    "plot_ts",
    "plot_ts_pred",
    "train_test_from_folds",
]

"""Quality / R-parity guards for feature selectors (step 0.23.x).

Paired against R on the identical rows (see ``daltoolbox/develop/parity/featsel*``),
comparing the SELECTED FEATURE SET. The deterministic selectors reproduce R exactly:

- corr, fss, ReliefF, info-gain, and stepwise with the gaussian family all match the
  R ``feature_selection_*`` selection on diabetes / breast_cancer.
- lasso (CV-chosen ``lambda``) and stepwise with the binomial family on the
  quasi-separable breast_cancer are inherently penalty-/solver-sensitive and only
  agree approximately; they are checked loosely (non-empty, mostly overlapping).

The R-verified sets are hard-coded here as the parity contract; the selectors are
native numpy / plain LinearRegression, so they are stable across backend versions.
"""
import warnings

import pandas as pd
from sklearn.datasets import load_breast_cancer

import pydaltoolbox as dal

warnings.filterwarnings("ignore")


def _breast():
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df = df.drop(columns="target")
    df["target"] = pd.Categorical.from_codes(raw.target, raw.target_names)
    return df


def _sel(model, data):
    return set(model.fit(data).selected)


# R-verified selected sets (featsel_samesplit.R) ---------------------------------
FSS_DIABETES = {"bmi", "bp", "s1", "s2", "s4", "s5", "s6", "sex"}
STEPWISE_GAUSS_DIABETES = {"bmi", "bp", "s1", "s2", "s5", "sex"}
RELIEF_TOP10_BREAST = {
    "mean_area", "mean_concave_points", "mean_perimeter", "mean_radius",
    "mean_texture", "worst_area", "worst_concave_points", "worst_perimeter",
    "worst_radius", "worst_texture",
}
INFOGAIN_TOP10_BREAST = {
    "area_error", "mean_area", "mean_concave_points", "mean_concavity",
    "mean_perimeter", "mean_radius", "worst_area", "worst_concave_points",
    "worst_perimeter", "worst_radius",
}


def test_corr_keeps_all_diabetes_features():
    """No diabetes pair exceeds |r|>0.9, so corr(0.9) drops nothing (matches R)."""
    diabetes = dal.datasets.diabetes()
    selected = _sel(dal.FeatureSelectionCorr(cutoff=0.9), diabetes)
    assert selected == set(diabetes.columns)


def test_fss_matches_r_on_diabetes():
    assert _sel(dal.FeatureSelectionFss("target"), dal.datasets.diabetes()) == FSS_DIABETES


def test_stepwise_gaussian_matches_r_on_diabetes():
    model = dal.FeatureSelectionStepwise("target", direction="forward", family="gaussian")
    assert _sel(model, dal.datasets.diabetes()) == STEPWISE_GAUSS_DIABETES


def test_relief_top10_matches_r_on_breast():
    model = dal.FeatureSelectionRelief("target", top=10, m=50, k=10, seed=1)
    assert _sel(model, _breast()) == RELIEF_TOP10_BREAST


def test_infogain_top10_matches_r_on_breast():
    assert _sel(dal.FeatureSelectionInfoGain("target", top=10), _breast()) == INFOGAIN_TOP10_BREAST


def test_lasso_and_binomial_stepwise_are_reasonable():
    """Penalty-/solver-sensitive selectors: only loose parity (non-empty, overlap)."""
    diabetes = dal.datasets.diabetes()
    lasso = _sel(dal.FeatureSelectionLasso("target"), diabetes)
    assert {"bmi", "bp", "s5"} <= lasso <= set(diabetes.columns) - {"target"}

    breast = _breast()
    step = _sel(dal.FeatureSelectionStepwise("target", direction="forward"), breast)
    assert len(step) >= 6  # converges on a parsimonious logistic subset

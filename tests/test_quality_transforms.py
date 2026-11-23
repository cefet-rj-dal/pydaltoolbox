"""Quality / R-parity guards for deterministic transforms (step 0.23.x).

Paired against R on the identical input (see ``daltoolbox/develop/parity/transforms*``);
transforms are deterministic, so the transformed OUTPUT MATRIX must match R up to float
noise. All eight checked transforms match exactly; two gaps were fixed to get there:

- ``CategMapping``: R ``categ_mapping`` returned only the dummy columns; it was aligned to
  the port (keep the other columns, replace the attribute by ``<attr>_<level>`` dummies).
- ``DtPca``: standardize with the *sample* SD (``ddof=1``) to match R ``prcomp(scale.=TRUE)``;
  ``StandardScaler`` (``ddof=0``) scaled the scores by ``sqrt((n-1)/n)`` (~0.011 off on iris).

These guards encode the fixes so neither can silently regress.
"""
import warnings

import numpy as np
import pytest

import pydaltoolbox as dal

warnings.filterwarnings("ignore")


def _iris():
    return dal.datasets.iris()


def test_minmax_scales_to_unit_range():
    out = dal.MinMax().fit(_iris()).transform(_iris()).select_dtypes("number")
    assert np.allclose(out.min(), 0.0) and np.allclose(out.max(), 1.0)


def test_zscore_uses_sample_sd():
    out = dal.ZScore().fit(_iris()).transform(_iris()).select_dtypes("number")
    assert np.allclose(out.mean(), 0.0, atol=1e-9)
    assert np.allclose(out.std(ddof=1), 1.0, atol=1e-9)  # sample SD, as in R


def test_categ_mapping_keeps_columns_and_names():
    iris = _iris()
    out = dal.CategMapping("Species").fit(iris).transform(iris)
    assert list(out.columns) == [
        "sepal_length", "sepal_width", "petal_length", "petal_width",
        "Species_setosa", "Species_versicolor", "Species_virginica",
    ]
    dummies = out[["Species_setosa", "Species_versicolor", "Species_virginica"]]
    assert (dummies.sum(axis=1) == 1).all()  # exactly one level per row


def test_dt_pca_matches_prcomp_scaling():
    iris = _iris()
    pca = dal.DtPca("Species", components=2).fit(iris)
    # sample SD (ddof=1), not population -- this is the prcomp(scale.=TRUE) convention
    raw = iris.drop(columns="Species").to_numpy(float)
    assert np.allclose(pca.scale_, raw.std(axis=0, ddof=1))
    # scores match R prcomp up to sign/float noise (R-verified first two rows)
    out = pca.transform(iris)
    got = out[["PC1", "PC2"]].abs().to_numpy()
    r_first_rows = np.array([[2.25714117564812, 0.4784238321249],
                             [2.07401301519962, 0.67188268702736]])
    assert got[:2] == pytest.approx(r_first_rows, abs=1e-9)

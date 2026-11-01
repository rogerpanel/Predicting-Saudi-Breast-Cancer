"""Guards for harmonisation and leakage errors (in-situ relabelling, ILC/IDC merging, duplicates,
leakage columns, out-of-fold preprocessing). Run: pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from bcgrade import evaluate as ev, harmonise as H, models as M  # noqa: E402

RAW = ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not (RAW / "metabric").exists(), reason="run scripts/download_tabular.py first")


@pytest.fixture(scope="module")
def cdm():
    df, flow = H.build_all(RAW)
    return df


# ---------------------------------------------------------------- harmonisation rules
@needs_data
def test_grade_values_are_invasive_nottingham_only(cdm):
    assert set(cdm.grade.dropna().unique()) <= {1.0, 2.0, 3.0}


@needs_data
def test_no_in_situ_relabelled_as_invasive(cdm):
    assert set(cdm["histo"].dropna().unique()) <= {"idc", "ilc", "mixed", "other"}
    raw = pd.read_csv(RAW / "broad" / "data_clinical_sample.txt", sep="\t", comment="#")
    dcis = set(raw.loc[raw.CANCER_TYPE_DETAILED.str.contains("In Situ", na=False), "PATIENT_ID"])
    assert dcis and not dcis & set(cdm.patient_id)


@needs_data
def test_lobular_not_merged_with_ductal(cdm):
    met = cdm[cdm.cohort == "METABRIC"]
    assert (met["histo"] == "ilc").sum() > 100 and (met["histo"] == "idc").sum() > 1000


@needs_data
def test_one_row_per_patient(cdm):
    assert not cdm.duplicated(["cohort", "patient_id"]).any()


def test_leakage_columns_never_in_feature_sets():
    for leak in ["npi", "tubule", "pleo", "mitotic", "grade", "stage", "cohort"]:
        assert leak not in H.FEATURES_FULL and leak not in H.FEATURES_CORE


def test_t_category_cutpoints():
    assert [H._t_from_size(x) for x in [5, 20, 20.1, 50, 51]] == [1, 1, 2, 2, 3]
    assert [H._n_from_code(x) for x in ["N0", "N0(i+)", "N1mi", "pN2a", "NX"]][:4] == [0, 0, 1, 1]


# ---------------------------------------------------------------- in-fold fitting
def test_imputer_and_scaler_are_fitted_inside_pipeline():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    X.iloc[::7, 1] = np.nan
    y = rng.integers(0, 3, 300)
    pipe, _ = M.make_model("mlr")
    tr = np.arange(150)
    pipe.fit(X.iloc[tr], y[tr])
    assert np.allclose(pipe.named_steps["scale"].mean_[0], X.iloc[tr, 0].mean(), atol=1e-8)
    assert not np.allclose(pipe.named_steps["scale"].mean_[0], X.iloc[:, 0].mean(), atol=1e-6)


def test_smote_only_in_training_step():
    pipe, _ = M.make_model("knn", smote=True)
    assert "smote" in pipe.named_steps
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(90, 3)))
    y = np.r_[np.zeros(60), np.ones(20), 2 * np.ones(10)].astype(int)
    pipe.fit(X, y)
    assert pipe.predict_proba(X).shape == (90, 3)  # predict never resamples


def test_ordinal_logit_recovers_direction():
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    lat = 1.5 * x + rng.logistic(size=2000)
    y = np.digitize(lat, [-1, 1])
    m = M.OrdinalLogit(alpha=1e-6).fit(x[:, None], y)
    assert 1.2 < m.coef_[0] < 1.8
    assert np.allclose(m.predict_proba(x[:, None]).sum(1), 1)


def test_balanced_rule_and_raking():
    P = np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4]])
    assert list(ev.balanced_rule(P, [0.8, 0.15, 0.05])) == [1, 2]
    d = pd.DataFrame({"a": [0, 0, 0, 1, 1, 0], "b": ["x", "y", "x", "y", "x", "y"]})
    w = ev.rake(d, {"a": {0: 0.5, 1: 0.5}, "b": {"x": 0.5, "y": 0.5}})
    assert abs(np.average(d.a == 1, weights=w) - 0.5) < 1e-6


def test_mi_classifier_pools_distinct_imputations():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    X.iloc[::4, 0] = np.nan
    y = (X.b + rng.normal(size=200) > 0).astype(int).to_numpy()
    mi = M.MIClassifier(LogisticRegression(), m=5, seed=1).fit(X, y)
    import copy
    imps = [copy.deepcopy(p).named_steps["impute"].transform(M.to_float_array(X))[::4, 0] for p in mi.pipes_]
    assert len({tuple(np.round(v, 6)) for v in imps}) == 5          # 5 distinct imputations
    P = np.mean([copy.deepcopy(p).predict_proba(X) for p in mi.pipes_], axis=0)
    assert np.allclose(mi.predict_proba(X), P)                       # predictions are averaged


@needs_data
def test_duke_excludes_in_situ_and_keeps_components_out(cdm):
    d = cdm[cdm.cohort == "Duke MRI"]
    assert len(d) > 900 and set(d.grade.dropna().unique()) <= {1.0, 2.0, 3.0}
    assert "tubule" not in H.FEATURES_FULL and "pleo" not in H.FEATURES_CORE

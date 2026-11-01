"""Map every open source onto one common data model (CDM).

CDM columns (one row per patient):
    cohort, patient_id, age, postmeno, er, pr, her2, hist, t_cat, n_pos, stage, grade
plus cohort-specific extras kept only for descriptive or leakage experiments:
    size_mm, nodes_pos, npi (METABRIC); tubule, pleo, mitotic (Sanger); country (Broad);
    race (TCGA); n_samples (MSK).

Harmonisation rules (see manuscript, Methods 2.3):
  * grade = invasive Nottingham / differentiation grade 1-3; anything else -> missing.
  * pure in-situ disease (DCIS/LCIS, METABRIC stage 0) and non-carcinomas are EXCLUDED, never
    relabelled as invasive.
  * hist in {idc, ilc, mixed, other}; IDC with a DCIS component is IDC; ILC is never merged with IDC.
  * t_cat: 1 (<=20 mm), 2 (>20-50 mm), 3 (>50 mm or T4 = ">=T3").
  * n_pos: any positive regional node (incl. micro-metastasis) -> 1; isolated tumour cells -> 0.
  * males excluded.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CDM = ["cohort", "patient_id", "age", "postmeno", "er", "pr", "her2", "histo",
       "t_cat", "n_pos", "stage", "grade"]
FEATURES_FULL = ["age", "postmeno", "er", "pr", "her2", "histo", "t_cat", "n_pos"]
FEATURES_CORE = ["age", "er", "pr", "her2", "histo"]


def _load(raw: Path, name: str) -> pd.DataFrame:
    s = pd.read_csv(raw / name / "data_clinical_sample.txt", sep="\t", comment="#", low_memory=False)
    p = pd.read_csv(raw / name / "data_clinical_patient.txt", sep="\t", comment="#", low_memory=False)
    return s.merge(p, on="PATIENT_ID", how="left", suffixes=("", "_pt"))


def _posneg(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"positive", "+ve", "pos", "positive low", "yes"}:
        return 1.0
    if s in {"negative", "-ve", "neg", "no"}:
        return 0.0
    return np.nan


def _roman_grade(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in {"1", "1.0"}:
        return 1.0
    if s in {"2", "2.0"}:
        return 2.0
    if s in {"3", "3.0"}:
        return 3.0
    m = re.match(r"^(III|II|I)\b", s)
    if m and "-" not in s.split()[0]:
        return float({"I": 1, "II": 2, "III": 3}[m.group(1)])
    return np.nan


def _t_from_size(mm):
    if pd.isna(mm):
        return np.nan
    return 1.0 if mm <= 20 else (2.0 if mm <= 50 else 3.0)


def _t_from_code(code):
    if pd.isna(code):
        return np.nan
    s = str(code).upper().replace(" ", "")
    if s.startswith("T1"):
        return 1.0
    if s.startswith("T2"):
        return 2.0
    if s.startswith("T3") or s.startswith("T4"):
        return 3.0
    return np.nan


def _n_from_code(code):
    if pd.isna(code):
        return np.nan
    s = str(code).upper().replace(" ", "")
    if s.startswith("P"):
        s = s[1:]
    if s.startswith("NX") or s in {"UNK", ""}:
        return np.nan
    if s.startswith("N0"):
        return 0.0
    if s[:2] in {"N1", "N2", "N3"}:
        return 1.0
    return np.nan


def _stage_roman(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().replace("STAGE", "").strip()
    for r, v in (("IV", 4), ("III", 3), ("II", 2), ("I", 1)):
        if s.startswith(r):
            return float(v)
    if s and s[0] in "1234":
        return float(s[0])
    return np.nan


def _hist_map(x, table):
    if pd.isna(x):
        return np.nan
    return table.get(str(x).strip(), "other")


class Log:
    """STROBE-style participant-flow log."""

    def __init__(self):
        self.rows = []

    def add(self, cohort, step, n):
        self.rows.append({"cohort": cohort, "step": step, "n": int(n)})

    def frame(self):
        return pd.DataFrame(self.rows)


# ----------------------------------------------------------------------------- cohorts
def metabric(raw, log):
    d = _load(raw, "metabric").drop_duplicates("PATIENT_ID")
    log.add("METABRIC", "records", len(d))
    d = d[d.SEX.eq("Female")]
    d = d[~d.CANCER_TYPE_DETAILED.str.contains("Angiosarcoma", na=False)]
    d = d[~d.TUMOR_STAGE.eq(0)]
    log.add("METABRIC", "invasive carcinoma, female (excl. stage 0 / sarcoma)", len(d))
    hist = {"Ductal/NST": "idc", "Lobular": "ilc", "Mixed": "mixed"}
    out = pd.DataFrame({
        "cohort": "METABRIC", "patient_id": d.PATIENT_ID,
        "age": d.AGE_AT_DIAGNOSIS,
        "postmeno": d.INFERRED_MENOPAUSAL_STATE.map({"Post": 1.0, "Pre": 0.0}),
        "er": d.ER_STATUS.map(_posneg), "pr": d.PR_STATUS.map(_posneg),
        "her2": d.HER2_STATUS.map(_posneg),
        "histo": d.HISTOLOGICAL_SUBTYPE.map(lambda x: _hist_map(x, hist)),
        "t_cat": d.TUMOR_SIZE.map(_t_from_size),
        "n_pos": (d.LYMPH_NODES_EXAMINED_POSITIVE > 0).astype(float).where(d.LYMPH_NODES_EXAMINED_POSITIVE.notna()),
        "stage": d.TUMOR_STAGE.astype(float),
        "grade": d.GRADE.map(_roman_grade),
        "size_mm": d.TUMOR_SIZE, "nodes_pos": d.LYMPH_NODES_EXAMINED_POSITIVE, "npi": d.NPI,
    })
    return out


def msk2018(raw, log):
    d = _load(raw, "msk2018")
    n_samples = d.groupby("PATIENT_ID").size()
    d = d.sort_values(["PATIENT_ID", "SAMPLE_TYPE"], ascending=[True, False])  # 'Primary' before 'Metastasis'
    log.add("MSK-2018", "samples", len(d))
    d = d.drop_duplicates("PATIENT_ID")
    log.add("MSK-2018", "patients", len(d))
    d = d[d.SEX.eq("Female")]
    log.add("MSK-2018", "female", len(d))
    hist = {"Breast Invasive Ductal Carcinoma": "idc", "Breast Invasive Lobular Carcinoma": "ilc",
            "Breast Mixed Ductal and Lobular Carcinoma": "mixed", "Breast Invasive Carcinoma, NOS": np.nan}
    out = pd.DataFrame({
        "cohort": "MSK-2018", "patient_id": d.PATIENT_ID,
        "age": d.INVASIVE_CARCINOMA_DX_AGE,
        "postmeno": d.MENOPAUSAL_STATUS_AT_DIAGNOSIS.map({"Post": 1.0, "Pre": 0.0, "Peri": 0.0}),
        "er": d.ER_STATUS_PRIMARY.map(_posneg), "pr": d.PR_STATUS_PRIMARY.map(_posneg),
        "her2": d.HER2_STATUS_PRIMARY.map(_posneg),
        "histo": d.TUMOR_SAMPLE_HISTOLOGY.map(lambda x: _hist_map(x, hist)),
        "t_cat": d.T_STAGE.map(_t_from_code), "n_pos": d.N_STAGE.map(_n_from_code),
        "stage": d.STAGE_AT_DIAGNOSIS.map(_stage_roman),
        "grade": d.OVERALL_TUMOR_GRADE.map(_roman_grade),
        "n_samples": d.PATIENT_ID.map(n_samples),
    })
    return out


def mbcproject(raw, log):
    d = _load(raw, "mbcproject").drop_duplicates("PATIENT_ID")
    log.add("MBCproject", "patients", len(d))
    d = d[d.SEX.eq("FEMALE")]
    log.add("MBCproject", "female, abstraction complete", len(d))
    d = d[~d.DX_HISTOLOGY.eq("DCIS")]
    log.add("MBCproject", "invasive carcinoma", len(d))

    def age_mid(s):
        if pd.isna(s):
            return np.nan
        s = str(s)
        if s.startswith("<"):
            return float(s[1:]) - 2
        a, b = s.split("-")
        return (float(a) + float(b)) / 2

    hist = {"IDC": "idc", "ILC": "ilc", "MIXED_IDLC": "mixed", "NOT_FOUND_IN_RECORD": np.nan}
    out = pd.DataFrame({
        "cohort": "MBCproject", "patient_id": d.PATIENT_ID,
        "age": d.AGE_AT_DIAGNOSIS.map(age_mid),
        "postmeno": d.MENOPAUSAL_STATUS_DX.map({"POSTMENOPAUSAL": 1.0, "PREMENOPAUSAL": 0.0, "PERIMENOPAUSAL": 0.0}),
        "er": d.DX_ER.map(_posneg), "pr": d.DX_PR.map(_posneg), "her2": d.DX_HER2OVERALL.map(_posneg),
        "histo": d.DX_HISTOLOGY.map(lambda x: _hist_map(x, hist)),
        "t_cat": np.nan, "n_pos": np.nan,
        "stage": d.DX_STAGE.map(_stage_roman),
        "grade": d.DX_GRADE.map(_roman_grade),
    })
    return out


def sanger(raw, log):
    d = _load(raw, "sanger").drop_duplicates("PATIENT_ID")
    log.add("Sanger", "patients", len(d))
    hist = {"Ductal": "idc", "Lobular": "ilc", "Lobular Var": "ilc", "Classic-alveolar lobular": "ilc",
            "Mixed": "mixed", "Ductal/Lobular": "mixed", "Unavailable": np.nan, "Unknown": np.nan}
    return pd.DataFrame({
        "cohort": "Sanger", "patient_id": d.PATIENT_ID, "age": d.AGE.astype(float), "postmeno": np.nan,
        "er": d.ER_STATUS.map(_posneg), "pr": d.PR_STATUS.map(_posneg), "her2": d.HER2_STATUS.map(_posneg),
        "histo": d.HISTOPATHOLOGY.map(lambda x: _hist_map(x, hist)),
        "t_cat": np.nan, "n_pos": np.nan, "stage": np.nan, "grade": d.GRADE.map(_roman_grade),
        "tubule": d.TUBULE_SCORE, "pleo": d.PLEOMORPHISM_SCORE, "mitotic": d.MITOTIC_SCORE,
    })


def broad(raw, log):
    d = _load(raw, "broad").drop_duplicates("PATIENT_ID")
    log.add("Broad (MX/VN)", "patients", len(d))
    d = d[~d.CANCER_TYPE_DETAILED.str.contains("In Situ", na=False)]
    log.add("Broad (MX/VN)", "invasive carcinoma", len(d))
    hist = {"Infiltrating Ductal Carcinoma": "idc", "Infiltrating Lobular Carcinoma": "ilc",
            "Mixed Carcinoma": "mixed"}
    return pd.DataFrame({
        "cohort": "Broad (MX/VN)", "patient_id": d.PATIENT_ID, "age": d.AGE.astype(float),
        "postmeno": d.MENOPAUSE_STATUS.map({"Postmenopausal": 1.0, "Premenopausal": 0.0, "Perimenopausal": 0.0}),
        "er": d.ER_STATUS.map(_posneg), "pr": d.PR_STATUS.map(_posneg), "her2": d.HER2_STATUS.map(_posneg),
        "histo": d.CANCER_TYPE_DETAILED.map(lambda x: _hist_map(x, hist)),
        "t_cat": np.nan, "n_pos": np.nan, "stage": d.STAGE.map(_stage_roman),
        "grade": d.GRADE.map(_roman_grade), "country": d.COUNTRY,
    })


def bccrc(raw, log):
    d = _load(raw, "bccrc").drop_duplicates("PATIENT_ID")
    log.add("BCCRC (TNBC)", "patients", len(d))
    d = d[~d.HISTOLOGICAL_TYPE.eq("Phyllodes")]
    log.add("BCCRC (TNBC)", "carcinoma", len(d))
    hist = {"IDC": "idc", "Mixed": "mixed"}
    return pd.DataFrame({
        "cohort": "BCCRC (TNBC)", "patient_id": d.PATIENT_ID, "age": d.AGE.astype(float), "postmeno": np.nan,
        "er": d.ER_STATUS.map(_posneg), "pr": d.PR_STATUS.map(_posneg), "her2": d.IHC_HER2.map(_posneg),
        "histo": d.HISTOLOGICAL_TYPE.map(lambda x: _hist_map(x, hist)),
        "t_cat": (d.TUMOR_SIZE * 10).map(_t_from_size), "n_pos": d.CLIN_N_STAGE.map(_n_from_code),
        "stage": np.nan, "grade": d.GRADE.map(_roman_grade),
    })


def tcga(raw, log):
    d = _load(raw, "tcga").drop_duplicates("PATIENT_ID")
    log.add("TCGA-BRCA", "patients", len(d))
    d = d[d.SEX.eq("Female")]
    log.add("TCGA-BRCA", "female", len(d))
    hist = {"Breast Invasive Ductal Carcinoma": "idc", "Breast Invasive Lobular Carcinoma": "ilc",
            "Breast Invasive Carcinoma (NOS)": np.nan}
    return pd.DataFrame({
        "cohort": "TCGA-BRCA", "patient_id": d.PATIENT_ID, "age": d.AGE.astype(float), "postmeno": np.nan,
        "er": np.nan, "pr": np.nan, "her2": np.nan,
        "histo": d.CANCER_TYPE_DETAILED.map(lambda x: _hist_map(x, hist)),
        "t_cat": d.PATH_T_STAGE.map(_t_from_code), "n_pos": d.PATH_N_STAGE.map(_n_from_code),
        "stage": d.AJCC_PATHOLOGIC_TUMOR_STAGE.map(_stage_roman), "grade": np.nan, "race": d.RACE,
    })


def duke(raw, log):
    """Duke-Breast-Cancer-MRI (TCIA; CC BY-NC 4.0). Clinical sheet has a 3-row header
    (group / name / code book); columns are taken by position and checked by name."""
    f = raw / "duke" / "Clinical_and_Other_Features.xlsx"
    hdr = pd.read_excel(f, header=None, nrows=2).iloc[1].tolist()
    d = pd.read_excel(f, header=None, skiprows=3)
    pos = {"pid": 0, "dob": 19, "meno": 20, "er": 23, "pr": 24, "her2": 25, "t": 28, "n": 29,
           "tub": 31, "nuc": 32, "mit": 33, "grade": 34, "htype": 35}
    assert hdr[0] == "Patient ID" and hdr[34] == "Nottingham grade" and hdr[35] == "Histologic type"
    d = d[list(pos.values())].set_axis(list(pos), axis=1).drop_duplicates("pid")
    log.add("Duke MRI", "patients", len(d))
    d = d[~d.htype.isin([0, 4])]   # pure DCIS / LCIS excluded, never relabelled
    log.add("Duke MRI", "invasive carcinoma", len(d))
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    hist = {1: "idc", 2: "ilc", 6: "mixed", 3: "other", 5: "other", 7: "other", 8: "other", 9: "other", 10: "other"}
    t, n = num(d.t), num(d.n)
    return pd.DataFrame({
        "cohort": "Duke MRI", "patient_id": d.pid, "age": -num(d.dob) / 365.25,
        "postmeno": num(d.meno).map({1: 1.0, 0: 0.0}),
        "er": num(d.er).map({1: 1.0, 0: 0.0}), "pr": num(d.pr).map({1: 1.0, 0: 0.0}),
        "her2": num(d.her2).map({1: 1.0, 0: 0.0}),            # 2 = borderline -> missing
        "histo": num(d.htype).map(hist),
        "t_cat": t.map({1: 1.0, 2: 2.0, 3: 3.0, 4: 3.0}),       # -1 TX, 0 T0, 5 Tis -> missing
        "n_pos": n.map({0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}),       # -1 NX -> missing
        "stage": np.nan, "grade": num(d.grade).where(num(d.grade).isin([1, 2, 3])),
        "tubule": num(d.tub), "pleo": num(d.nuc), "mitotic": num(d.mit),
    })


def duke_radiomics(raw):
    """529 pre-computed DCE-MRI features (Saha et al. 2018), one row per patient."""
    r = pd.read_excel(raw / "duke" / "Imaging_Features.xlsx")
    return r.rename(columns={r.columns[0]: "patient_id"}).set_index("patient_id")


def ljubljana(raw):
    cols = ["age", "menopause", "tumor_size", "inv_nodes", "node_caps", "deg_malig", "breast",
            "breast_quad", "irradiat", "recurrence"]
    d = pd.read_csv(raw / "ljubljana" / "breast-cancer.csv", header=None, names=cols, quotechar="'",
                    na_values=["?"])
    lo = lambda s: float(str(s).split("-")[0]) + 2 if pd.notna(s) else np.nan  # noqa: E731
    return pd.DataFrame({
        "age": d.age.map(lambda s: float(str(s).split("-")[0]) + 5 if pd.notna(s) else np.nan),
        "postmeno": d.menopause.map({"ge40": 1.0, "lt40": 1.0, "premeno": 0.0}),
        "size_mm": d.tumor_size.map(lo),
        "n_pos": d.inv_nodes.map(lambda s: 0.0 if str(s) == "0-2" else 1.0),
        "node_caps": d.node_caps.map({"yes": 1.0, "no": 0.0}),
        "grade": d.deg_malig.astype(float),
    })


GRADED = [metabric, msk2018, mbcproject, sanger, broad, bccrc, duke]


def build_all(raw: Path):
    log = Log()
    frames = [f(raw, log) for f in GRADED] + [tcga(raw, log)]
    allc = pd.concat(frames, ignore_index=True)
    for c, g in allc.groupby("cohort"):
        if g.grade.notna().any():
            log.add(c, "analysis set (grade 1-3 recorded)", g.grade.notna().sum())
    return allc, log.frame()

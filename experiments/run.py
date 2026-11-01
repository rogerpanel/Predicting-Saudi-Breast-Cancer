#!/usr/bin/env python3
"""Run experiments E1-E8. Results go to results/ (CSV/JSON/NPZ); report.py turns them into
LaTeX tables and figures.

    python experiments/run.py all            # everything
    python experiments/run.py e2 e3          # selected stages
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bcgrade import describe, evaluate as ev, harmonise as H, models as M  # noqa: E402

warnings.filterwarnings("ignore")
CFG = yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))
SEED = CFG["seed"]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    p = ROOT / "data" / "processed" / "cdm.csv"
    if not p.exists():
        df, flow = H.build_all(ROOT / "data" / "raw")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        flow.to_csv(RES / "e1_flow.csv", index=False)
    return pd.read_csv(p, low_memory=False)


def xy(df, feats):
    d = df[df.grade.notna()]
    return M.design(d, feats), (d.grade.to_numpy() - 1).astype(int), d


def fit_tuned(name, X, y, seed, smote=False, inner=None):
    pipe, grid = M.make_model(name, seed, smote)
    if not grid:
        return clone(pipe).fit(X, y), {}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(inner or CFG["cv"]["inner_folds"], shuffle=True,
                                                     random_state=seed),
                      scoring=CFG["cv"]["scoring"], n_jobs=2, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def prior_of(y):
    return np.bincount(y, minlength=3) / len(y)


def optional_models():
    """Optional models (TabPFN) are used only if they can actually be fitted here; status is logged."""
    ok, status = [], {}
    for name in CFG.get("optional_models", []):
        try:
            rng = np.random.default_rng(0)
            Xs = pd.DataFrame(rng.normal(size=(60, 3)))
            M.make_model(name, SEED)[0].fit(Xs, np.arange(60) % 3)
            ok.append(name)
            status[name] = "available"
        except Exception as e:  # noqa: BLE001
            status[name] = f"not run: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    json.dump(status, open(RES / "optional_models_status.json", "w"), indent=2)
    return ok


def models_run():
    return CFG["ladder"] + [m for m in CFG.get("optional_models", []) if (RES / f"e2_oof_{m}.npz").exists()]


def add_delta(row, y, P, Pref, key="d"):
    d = ev.paired_delta(y, P, Pref, B=CFG["bootstrap"], seed=SEED)
    for k, (e, lo, hi, p) in d.items():
        row.update({f"{key}_{k}": e, f"{key}_{k}_lo": lo, f"{key}_{k}_hi": hi, f"{key}_{k}_p": p})
    return row


# ----------------------------------------------------------------------------- E1
def e1(df):
    log("E1 descriptive + associations")
    rows = []
    for c, g in df.groupby("cohort", sort=False):
        gg = g[g.grade.notna()] if g.grade.notna().any() else g
        r = {"cohort": c, "n": len(gg), "age_median": gg.age.median(),
             "age_q1": gg.age.quantile(.25), "age_q3": gg.age.quantile(.75),
             "age_le50": (gg.age <= 50).mean() if gg.age.notna().any() else np.nan}
        for v in ["er", "pr", "her2", "postmeno", "n_pos"]:
            r[v] = gg[v].mean()
            r[v + "_miss"] = gg[v].isna().mean()
        for k in [1, 2, 3]:
            r[f"t{k}"] = (gg.t_cat == k).sum() / gg.t_cat.notna().sum() if gg.t_cat.notna().any() else np.nan
            r[f"g{k}"] = (gg.grade == k).mean() if gg.grade.notna().any() else np.nan
        for h in ["idc", "ilc", "mixed", "other"]:
            r[h] = (gg["histo"] == h).sum() / gg["histo"].notna().sum() if gg["histo"].notna().any() else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "e1_cohorts.csv", index=False)
    at = describe.association_table(df, ["METABRIC", "MSK-2018"], B=CFG["bootstrap"], seed=SEED)
    at.to_csv(RES / "e1_associations.csv", index=False)


# ----------------------------------------------------------------------------- E2
def e2(df, models=None, append=False):
    log("E2 nested repeated CV on METABRIC (full CDM features)")
    X, y, _ = xy(df[df.cohort == CFG["dev_cohort"]], H.FEATURES_FULL)
    R, K = CFG["cv"]["repeats"], CFG["cv"]["outer_folds"]
    out, per_rep, chosen = {}, [], []
    for name in (models if models is not None else CFG["ladder"] + optional_models()):
        t0 = time.time()
        P = np.zeros((R, len(y), 3))
        PR = np.zeros((R, len(y), 3))
        for r in range(R):
            skf = StratifiedKFold(K, shuffle=True, random_state=SEED + r)
            for f, (tr, te) in enumerate(skf.split(X, y)):
                est, bp = fit_tuned(name, X.iloc[tr], y[tr], SEED + r)
                P[r, te] = est.predict_proba(X.iloc[te])
                PR[r, te] = prior_of(y[tr])
                chosen.append({"model": name, "repeat": r, "fold": f, **{k: str(v) for k, v in bp.items()}})
            m = ev.metrics(y, P[r], PR[r])
            per_rep.append({"model": name, "repeat": r, **m})
        out[name] = P
        np.savez_compressed(RES / f"e2_oof_{name}.npz", P=P, PR=PR, y=y)
        log(f"   {name:9s} macroAUC={np.mean([d['macro_auc'] for d in per_rep if d['model']==name]):.3f}"
            f"  ({time.time()-t0:.0f}s)")
    hp = pd.DataFrame(chosen)
    if append and (RES / "e2_hyperparams.csv").exists():
        old = pd.read_csv(RES / "e2_hyperparams.csv")
        hp = pd.concat([old[~old.model.isin(hp.model.unique())], hp], ignore_index=True)
    hp.to_csv(RES / "e2_hyperparams.csv", index=False)
    e2s(df)


def e2opt(df):
    """Add optional models (TabPFN) to E2 without re-running the primary ladder (needs HF_TOKEN)."""
    opt = optional_models()
    if not opt:
        log("   no optional model available -- see results/optional_models_status.json")
        return
    e2(df, models=opt, append=True)


def e2s(df):
    """Summaries for E2 from the saved out-of-fold predictions (re-runnable without refitting)."""
    log("E2 summary (per-repeat metrics + bootstrap CIs of repeat-averaged OOF predictions)")
    per_rep, summ = [], []
    Pref = np.load(RES / "e2_oof_mlr.npz")["P"].mean(0)
    for name in models_run():
        z = np.load(RES / f"e2_oof_{name}.npz")
        y = z["y"]
        for r in range(z["P"].shape[0]):
            per_rep.append({"model": name, "repeat": r, **ev.metrics(y, z["P"][r], z["PR"][r])})
        Pm, prior = z["P"].mean(0), z["PR"].mean(0)
        pr = pd.DataFrame([d for d in per_rep if d["model"] == name])
        s = pr.drop(columns=["model", "repeat"]).mean().to_dict()
        ci = ev.bootstrap(y, Pm, prior, B=CFG["bootstrap"], seed=SEED)
        s.update({f"{k}_lo": v[0] for k, v in ci.items()})
        s.update({f"{k}_hi": v[1] for k, v in ci.items()})
        s["model"] = name
        s["auc_rep_sd"] = pr.macro_auc.std()
        s["macro_auc_avgoof"] = ev.fast_metrics(y, Pm, prior)["macro_auc"]
        add_delta(s, y, Pm, Pref)
        summ.append(s)
    pd.DataFrame(per_rep).to_csv(RES / "e2_per_repeat.csv", index=False)
    pd.DataFrame(summ).to_csv(RES / "e2_summary.csv", index=False)


# ----------------------------------------------------------------------------- E3
def e3(df):
    log("E3 external validation + LODO")
    dev = df[df.cohort == CFG["dev_cohort"]]
    rows, store = [], {}
    for fs_name, feats, targets in [("full", H.FEATURES_FULL, CFG["external_full"]),
                                    ("core", H.FEATURES_CORE, CFG["external_core"])]:
        X, y, _ = xy(dev, feats)
        prior = prior_of(y)
        for name in ["mlr"] + [m for m in models_run() if m != "mlr"]:
            est, bp = fit_tuned(name, X, y, SEED, inner=5)
            for tgt in targets:
                Xt, yt, _ = xy(df[df.cohort == tgt], feats)
                P = est.predict_proba(Xt)
                m = ev.metrics(yt, P, prior)
                ci = ev.bootstrap(yt, P, prior, B=CFG["bootstrap"], seed=SEED)
                m.update({f"{k}_lo": v[0] for k, v in ci.items()})
                m.update({f"{k}_hi": v[1] for k, v in ci.items()})
                add_delta(m, yt, P, store[f"{fs_name}|mlr|{tgt}"][1]) if name != "mlr" else None
                rows.append({"design": "METABRIC->ext", "features": fs_name, "model": name, "target": tgt,
                             "n": len(yt), **m, "params": json.dumps({k: str(v) for k, v in bp.items()})})
                store[f"{fs_name}|{name}|{tgt}"] = (yt, P, prior)
    # LODO
    cohorts = CFG["lodo_cohorts"]
    for held in cohorts:
        tr = df[df.cohort.isin([c for c in cohorts if c != held])]
        X, y, _ = xy(tr, H.FEATURES_CORE)
        Xt, yt, _ = xy(df[df.cohort == held], H.FEATURES_CORE)
        prior = prior_of(y)
        for name in CFG["lodo_models"]:
            est, _ = fit_tuned(name, X, y, SEED)
            P = est.predict_proba(Xt)
            m = ev.metrics(yt, P, prior)
            ci = ev.bootstrap(yt, P, prior, B=CFG["bootstrap"], seed=SEED)
            m.update({f"{k}_lo": v[0] for k, v in ci.items()})
            m.update({f"{k}_hi": v[1] for k, v in ci.items()})
            rows.append({"design": "LODO", "features": "core", "model": name, "target": held, "n": len(yt), **m})
            store[f"lodo|{name}|{held}"] = (yt, P, prior)
    pd.DataFrame(rows).to_csv(RES / "e3_external.csv", index=False)
    np.savez_compressed(RES / "e3_preds.npz", **{k: np.concatenate([v[0][:, None], v[1]], 1)
                                                 for k, v in store.items()})


# ----------------------------------------------------------------------------- E4
def _cv_auc(X, y, name, seed, groups=None, fixed=None, folds=5):
    """One 5-fold CV; returns OOF macro-AUC and G3-AUC."""
    P = np.zeros((len(y), 3))
    PR = np.zeros((len(y), 3))
    splitter = (GroupKFold(folds) if groups is not None else StratifiedKFold(folds, shuffle=True, random_state=seed))
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        if fixed is not None:
            est = clone(fixed).fit(X.iloc[tr], y[tr])
        else:
            est, _ = fit_tuned(name, X.iloc[tr], y[tr], seed)
        P[te] = est.predict_proba(X.iloc[te])
        PR[te] = prior_of(y[tr])
    m = ev.metrics(y, P, PR, calib=False)
    return m["macro_auc"], m["g3_auc"], m["macro_f1"]


def e4(df):
    log("E4 leakage audit")
    rows = []
    met = df[df.cohort == "METABRIC"]
    # (a) target leakage: NPI = 0.2*size_cm + node stage + GRADE
    for name in ["mlr", "xgb"]:
        X, y, d = xy(met, H.FEATURES_FULL)
        honest = _cv_auc(X, y, name, SEED)
        X2 = X.copy()
        X2["npi"] = d.npi.to_numpy()
        leaky = _cv_auc(X2, y, name, SEED)
        rows.append({"scenario": "Target leakage: NPI as predictor (METABRIC)", "model": name,
                     "honest_auc": honest[0], "leaky_auc": leaky[0], "honest_f1": honest[2], "leaky_f1": leaky[2]})
    # (b) SMOTE before the split
    for name in ["knn", "xgb"]:
        X, y, _ = xy(met, H.FEATURES_FULL)
        honest = _cv_auc(X, y, name, SEED, fixed=M.make_model(name, SEED, smote=True)[0].set_params(
            **({"clf__n_neighbors": 5} if name == "knn" else {"clf__max_depth": 4})))
        from imblearn.over_sampling import SMOTE
        Xa = M.to_float_array(X)
        Xi = pd.DataFrame(M._imputer(SEED).fit_transform(Xa), columns=X.columns)
        Xs, ys = SMOTE(random_state=SEED).fit_resample(Xi, y)
        fixed = M.make_model(name, SEED)[0].set_params(
            **({"clf__n_neighbors": 5} if name == "knn" else {"clf__max_depth": 4}))
        leaky = _cv_auc(Xs.reset_index(drop=True), ys, name, SEED, fixed=fixed)
        rows.append({"scenario": "SMOTE + imputation before split (METABRIC)", "model": name,
                     "honest_auc": honest[0], "leaky_auc": leaky[0], "honest_f1": honest[2], "leaky_f1": leaky[2]})
    # (c) Nottingham component scores as predictors (Sanger)
    san = df[df.cohort == "Sanger"]
    X, y, d = xy(san, H.FEATURES_CORE)
    fixed = M.make_model("mlr", SEED)[0]
    honest = _cv_auc(X, y, "mlr", SEED, fixed=fixed)
    X2 = X.copy()
    for c in ["tubule", "pleo", "mitotic"]:
        X2[c] = d[c].to_numpy()
    leaky = _cv_auc(X2, y, "mlr", SEED, fixed=fixed)
    rows.append({"scenario": "Grade components as predictors (Sanger)", "model": "mlr",
                 "honest_auc": honest[0], "leaky_auc": leaky[0], "honest_f1": honest[2], "leaky_f1": leaky[2]})
    duk = df[df.cohort == "Duke MRI"]
    X, y, d = xy(duk, H.FEATURES_FULL)
    honest = _cv_auc(X, y, "mlr", SEED, fixed=fixed)
    X2 = X.copy()
    for c in ["tubule", "pleo", "mitotic"]:
        X2[c] = d[c].to_numpy()
    leaky = _cv_auc(X2, y, "mlr", SEED, fixed=fixed)
    rows.append({"scenario": "Grade components as predictors (Duke MRI)", "model": "mlr",
                 "honest_auc": honest[0], "leaky_auc": leaky[0], "honest_f1": honest[2], "leaky_f1": leaky[2]})
    # (d) record-level vs patient-level splitting (MSK: several sequenced samples per patient)
    msk = df[(df.cohort == "MSK-2018") & df.grade.notna()]
    rep = msk.loc[msk.index.repeat(msk.n_samples.fillna(1).astype(int))].reset_index(drop=True)
    X, y, d = xy(rep, H.FEATURES_FULL)
    groups = d.patient_id.to_numpy()
    for name, params in [("knn", {"clf__n_neighbors": 1}), ("rf", {"clf__min_samples_leaf": 1})]:
        fixed = M.make_model(name, SEED)[0].set_params(**params)
        leaky = _cv_auc(X, y, name, SEED, fixed=fixed)
        honest = _cv_auc(X, y, name, SEED, groups=groups, fixed=fixed)
        rows.append({"scenario": f"Record-level split, {len(rep)} records / {msk.shape[0]} patients (MSK)",
                     "model": name, "honest_auc": honest[0], "leaky_auc": leaky[0],
                     "honest_f1": honest[2], "leaky_f1": leaky[2]})
    pd.DataFrame(rows).to_csv(RES / "e4_leakage.csv", index=False)


# ----------------------------------------------------------------------------- E5
def e5(df):
    log("E5 calibration, recalibration, decision curves")
    th = np.round(np.arange(0.05, 0.86, 0.01), 2)
    curves, dca, cal, oe = [], [], [], []
    _, y, _ = xy(df[df.cohort == "METABRIC"], H.FEATURES_FULL)
    preds = np.load(RES / "e3_preds.npz")
    for name in ["mlr", "ordinal", "xgb"]:
        Pm = np.load(RES / f"e2_oof_{name}.npz")["P"].mean(0)
        sets = [("METABRIC (internal CV)", (y == 2).astype(int), Pm[:, 2])]
        oe.append({"model": name, "set": "METABRIC (internal CV)", **{f"obs_g{k+1}": np.mean(y == k) for k in range(3)},
                   **{f"pred_g{k+1}": Pm[:, k].mean() for k in range(3)}})
        for tgt in CFG["external_full"]:
            a = preds[f"full|{name}|{tgt}"]
            yt, Pt = a[:, 0].astype(int), a[:, 1:]
            oe.append({"model": name, "set": f"{tgt} (external)", **{f"obs_g{k+1}": np.mean(yt == k) for k in range(3)},
                       **{f"pred_g{k+1}": Pt[:, k].mean() for k in range(3)}})
            y3 = (yt == 2).astype(int)
            sets.append((f"{tgt} (external)", y3, Pt[:, 2]))
            # intercept-only recalibration, 2-fold cross-fitted inside the external cohort
            lp = np.log(np.clip(Pt[:, 2], 1e-6, 1 - 1e-6) / (1 - np.clip(Pt[:, 2], 1e-6, 1 - 1e-6)))
            rec = np.zeros_like(lp)
            import statsmodels.api as sm
            for tr, te in StratifiedKFold(2, shuffle=True, random_state=SEED).split(lp, y3):
                a0 = sm.GLM(y3[tr], np.ones((len(tr), 1)), family=sm.families.Binomial(), offset=lp[tr]).fit().params[0]
                rec[te] = 1 / (1 + np.exp(-(lp[te] + a0)))
            sets.append((f"{tgt} (intercept-updated)", y3, rec))
        for label, yy, pp in sets:
            ci, cs = ev.cal_intercept_slope(yy, pp)
            from sklearn.metrics import roc_auc_score
            cal.append({"model": name, "set": label, "n": len(yy), "prev": yy.mean(), "mean_pred": pp.mean(),
                        "cal_int": ci, "cal_slope": cs, "auc": roc_auc_score(yy, pp),
                        "brier": np.mean((pp - yy) ** 2)})
            xs, ys = ev.calibration_curve_lowess(yy, pp)
            curves.append(pd.DataFrame({"model": name, "set": label, "p": xs, "obs": ys}))
            nb = ev.net_benefit(yy, pp, th)
            nb["model_name"], nb["set"] = name, label
            dca.append(nb)
    pd.DataFrame(cal).to_csv(RES / "e5_calibration.csv", index=False)
    pd.DataFrame(oe).to_csv(RES / "e5_oe.csv", index=False)
    pd.concat(curves).to_csv(RES / "e5_curves.csv", index=False)
    pd.concat(dca).to_csv(RES / "e5_dca.csv", index=False)


# ----------------------------------------------------------------------------- E6
def e6(df):
    log("E6 SHAP plausibility + ordinal odds ratios")
    import shap
    from scipy import stats
    X, y, d = xy(df[df.cohort == "METABRIC"], H.FEATURES_FULL)
    est, _ = fit_tuned("xgb", X, y, SEED, inner=5)
    Xt = pd.DataFrame(est[:-1].transform(X), columns=X.columns)
    Xi = pd.DataFrame(est[:2].transform(X), columns=X.columns)  # to_array + impute (unscaled)
    sv = shap.TreeExplainer(est.named_steps["clf"]).shap_values(Xt)
    sv3 = sv[:, :, 2] if np.ndim(sv) == 3 else sv[2]
    exp = CFG["expected_direction"]
    rng = np.random.default_rng(SEED)
    rows = []
    for j, c in enumerate(X.columns):
        obs = X[c].notna().to_numpy()
        xv, s = Xi[c].to_numpy()[obs], sv3[obs, j]
        rho = stats.spearmanr(xv, s)[0] if np.std(xv) > 0 else np.nan
        bs = []
        for _ in range(500):
            i = rng.integers(0, len(s), len(s))
            bs.append(stats.spearmanr(xv[i], s[i])[0] if np.std(xv[i]) > 0 else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        e = exp.get(c)
        rows.append({"feature": c, "mean_abs_shap": np.abs(sv3[:, j]).mean(), "direction_rho": rho,
                     "rho_lo": lo, "rho_hi": hi, "expected": e,
                     "agrees": (None if e is None else bool(np.sign(rho) == e and (lo > 0 if e > 0 else hi < 0)))})
    pd.DataFrame(rows).to_csv(RES / "e6_shap.csv", index=False)
    np.savez_compressed(RES / "e6_shap_values.npz", sv3=sv3, X=Xi.to_numpy(), cols=np.array(X.columns))
    # ordinal (proportional-odds) model on complete cases, interpretable units
    cc = d[H.FEATURES_FULL].copy()
    cc["grade"] = y
    cc = cc.dropna()
    Xo = M.design(cc, H.FEATURES_FULL)
    Xo["age"] = Xo["age"] / 10.0
    yo = cc.grade.to_numpy()
    fit = M.OrdinalLogit(alpha=1e-6).fit(Xo.to_numpy(), yo)
    boots = []
    for _ in range(500):
        i = rng.integers(0, len(yo), len(yo))
        boots.append(M.OrdinalLogit(alpha=1e-6).fit(Xo.to_numpy()[i], yo[i]).coef_)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    pd.DataFrame({"term": Xo.columns, "beta": fit.coef_, "OR": np.exp(fit.coef_), "OR_lo": np.exp(lo),
                  "OR_hi": np.exp(hi), "n": len(yo)}).to_csv(RES / "e6_ordinal_or.csv", index=False)


# ----------------------------------------------------------------------------- E7
def _targets(anc, age_le50):
    """Raking targets from whatever marginals an anchor reports."""
    t = {}
    for v in ["er", "pr", "her2"]:
        if anc.get(f"{v}_pos") is not None:
            t[v] = {1.0: anc[f"{v}_pos"], 0.0: 1 - anc[f"{v}_pos"]}
    if anc.get("histology"):
        t["hist_r"] = dict(anc["histology"])
    a = anc.get("age_le50")
    a = age_le50 if a is None else a
    t["age_le50"] = {1.0: a, 0.0: 1 - a}
    return t


def _rake_frame(d):
    d = d.copy()
    d["hist_r"] = d["histo"].replace({"mixed": "idc"})
    d["age_le50"] = (d.age <= 50).astype(float).where(d.age.notna())
    return d


def e7(df):
    log("E7 Saudi case-mix transport (all anchors, pooled anchor, age sensitivity, bootstrap CIs)")
    anchors = yaml.safe_load(open(ROOT / "configs" / "anchors.yaml"))
    rake_anchors = {k: v for k, v in anchors.items() if k.startswith("saudi_") and v.get("er_pos") is not None}
    preds = np.load(RES / "e3_preds.npz")
    prior_dev = prior_of(np.load(RES / "e2_oof_mlr.npz")["y"])
    rng = np.random.default_rng(SEED)
    rows, sens = [], []
    for akey, anc in rake_anchors.items():
        tg = _targets(anc, CFG["age_le50_default"])
        for cohort in CFG["transport_cohorts"]:
            _, y, d = xy(df[df.cohort == cohort], H.FEATURES_FULL)
            d = _rake_frame(d)
            w = ev.rake(d, tg)
            inc = w > 0
            # bootstrap CI of the re-weighted grade-3 share (resample patients, re-rake)
            bs = []
            idx = np.where(inc)[0]
            for _ in range(CFG["transport_boot"]):
                i = rng.choice(idx, len(idx))
                wb = ev.rake(d.iloc[i].reset_index(drop=True), tg)
                bs.append(np.average(y[i] == 2, weights=wb) if wb.sum() > 0 else np.nan)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            base = {"anchor": akey, "anchor_label": anc["label"], "verified": anc["verified"], "cohort": cohort,
                    "raked_on": "+".join(tg), "n_raked": int(inc.sum()), "ess": ev.ess(w), "w_max": w.max(),
                    "rw_g3_lo": lo, "rw_g3_hi": hi}
            for k in range(3):
                base[f"obs_g{k+1}"] = np.mean(y[inc] == k)
                base[f"rw_g{k+1}"] = np.average(y[inc] == k, weights=w[inc])
                base[f"saudi_g{k+1}"] = anc["grade"][k + 1]
            for name in ["mlr", "ordinal", "xgb"]:
                if cohort == CFG["dev_cohort"]:
                    z = np.load(RES / f"e2_oof_{name}.npz")
                    P, prior = z["P"].mean(0), z["PR"].mean(0)[inc]
                else:
                    P = preds[f"full|{name}|{cohort}"][:, 1:]
                    prior = prior_dev
                m0 = ev.metrics(y[inc], P[inc], prior, calib=False)
                m1 = ev.metrics(y[inc], P[inc], prior, weights=w[inc], calib=False)
                r = dict(base, model=name, auc_unw=m0["macro_auc"], auc_w=m1["macro_auc"], f1_unw=m0["macro_f1"],
                         f1_w=m1["macro_f1"], g3auc_unw=m0["g3_auc"], g3auc_w=m1["g3_auc"])
                for k in range(3):
                    r[f"pred_g{k+1}"] = np.average(P[inc, k], weights=w[inc])
                rows.append(r)
            if anc.get("age_le50") is None and cohort == CFG["dev_cohort"]:
                for a in CFG["age_le50_sensitivity"]:
                    wa = ev.rake(d, _targets(anc, a))
                    ia = wa > 0
                    sens.append({"anchor": akey, "cohort": cohort, "age_le50": a, "ess": ev.ess(wa),
                                 **{f"rw_g{k+1}": np.average(y[ia] == k, weights=wa[ia]) for k in range(3)}})
    pd.DataFrame(rows).to_csv(RES / "e7_transport.csv", index=False)
    pd.DataFrame(sens).to_csv(RES / "e7_age_sensitivity.csv", index=False)
    # random-effects pooled published Saudi grade-3 (and grade-1) share across all anchors reporting it
    main = {k: v for k, v in anchors.items() if v.get("grade_counts") and v.get("use") != "pooled_sensitivity_only"}
    allg = {k: v for k, v in anchors.items() if v.get("grade_counts")}
    cnt = lambda d, g: [(k, v["grade_counts"][g], sum(v["grade_counts"].values())) for k, v in d.items()]  # noqa: E731
    pool = []
    for lab, lst in [("g3", cnt(main, 3)), ("g1", cnt(main, 1)), ("g3_all", cnt(allg, 3))]:
        p, lo, hi, tau2 = ev.dl_pool_logit([e for _, e, _ in lst], [n for _, _, n in lst])
        pool.append({"quantity": lab, "k_studies": len(lst), "n_total": int(sum(n for _, _, n in lst)), "pooled": p,
                     "lo": lo, "hi": hi, "tau2": tau2, "studies": ";".join(k for k, _, _ in lst)})
    pd.DataFrame(pool).to_csv(RES / "e7_pooled_saudi.csv", index=False)


# ----------------------------------------------------------------------------- E8
def e8(df):
    log("E8 Ljubljana sensitivity ladder")
    lj = H.ljubljana(ROOT / "data" / "raw").dropna(subset=["grade"])
    feats = ["age", "postmeno", "size_mm", "n_pos", "node_caps"]
    X, y = lj[feats], (lj.grade.to_numpy() - 1).astype(int)
    rows = []
    for name in ["majority", "mlr", "ordinal", "rf", "xgb"]:
        aucs = []
        for r in range(CFG["cv"]["repeats"]):
            P = np.zeros((len(y), 3))
            PR = np.zeros((len(y), 3))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
                est, _ = fit_tuned(name, X.iloc[tr], y[tr], SEED + r)
                P[te] = est.predict_proba(X.iloc[te])
                PR[te] = prior_of(y[tr])
            aucs.append(ev.metrics(y, P, PR, calib=False))
        a = pd.DataFrame(aucs)
        rows.append({"model": name, "n": len(y), "macro_auc": a.macro_auc.mean(), "macro_auc_sd": a.macro_auc.std(),
                     "macro_f1": a.macro_f1.mean(), "g3_auc": a.g3_auc.mean()})
    pd.DataFrame(rows).to_csv(RES / "e8_ljubljana.csv", index=False)


# ----------------------------------------------------------------------------- E2mi
def e2mi(df):
    """RUNBOOK R4: multiple imputation (m draws, pooled predictions) vs single imputation vs complete case.
    Hyper-parameters are fixed at the modal E2 choice to keep m x folds x repeats tractable."""
    log("E2mi multiple imputation + complete-case sensitivity")
    X, y, _ = xy(df[df.cohort == CFG["dev_cohort"]], H.FEATURES_FULL)
    hp = pd.read_csv(RES / "e2_hyperparams.csv")
    R, K, m = CFG["cv"]["repeats"], CFG["cv"]["outer_folds"], CFG["mi"]["m"]
    from joblib import Parallel, delayed
    rows = []
    for name in CFG["mi"]["models"]:
        h = hp[hp.model == name].drop(columns=["model", "repeat", "fold"]).dropna(axis=1, how="all")
        def _val(v):
            v = yaml.safe_load(v) if isinstance(v, str) else v
            return int(v) if isinstance(v, float) and v.is_integer() else v
        params = {c: _val(h[c].mode().iloc[0]) for c in h.columns}
        clf = M.base_estimator(name, SEED, params)

        def one(r, tr, te):
            est = M.MIClassifier(clf, m=m, seed=SEED + 1000 * r).fit(X.iloc[tr], y[tr])
            return te, est.predict_proba(X.iloc[te]), prior_of(y[tr])

        P = np.zeros((R, len(y), 3))
        PR = np.zeros((R, len(y), 3))
        for r in range(R):
            res = Parallel(n_jobs=4)(delayed(one)(r, tr, te) for tr, te in
                                     StratifiedKFold(K, shuffle=True, random_state=SEED + r).split(X, y))
            for te, p, pr in res:
                P[r, te], PR[r, te] = p, pr
        Pm, prior = P.mean(0), PR.mean(0)
        mm = pd.DataFrame([ev.metrics(y, P[r], PR[r]) for r in range(R)]).mean().to_dict()
        ci = ev.bootstrap(y, Pm, prior, B=CFG["bootstrap"], seed=SEED)
        zs = np.load(RES / f"e2_oof_{name}.npz")
        si = pd.DataFrame([ev.metrics(zs["y"], zs["P"][r], zs["PR"][r]) for r in range(zs["P"].shape[0])]).mean().to_dict()
        row = {"model": name, "method": f"Multiple imputation (m={m}, pooled predictions)", "n": len(y), **mm,
               "macro_auc_lo": ci["macro_auc"][0], "macro_auc_hi": ci["macro_auc"][1],
               "params": json.dumps({k: str(v) for k, v in params.items()}), "si_macro_auc": si["macro_auc"]}
        add_delta(row, y, Pm, zs["P"].mean(0))
        rows.append(row)
        np.savez_compressed(RES / f"e2mi_oof_{name}.npz", P=P, PR=PR, y=y)
        # complete-case sensitivity (tuned, same nested design, fewer patients)
        cc = X.notna().all(axis=1).to_numpy()
        Xc, yc = X[cc].reset_index(drop=True), y[cc]
        Pc = np.zeros((R, len(yc), 3))
        PRc = np.zeros((R, len(yc), 3))
        for r in range(R):
            for tr, te in StratifiedKFold(K, shuffle=True, random_state=SEED + r).split(Xc, yc):
                est, _ = fit_tuned(name, Xc.iloc[tr], yc[tr], SEED + r)
                Pc[r, te], PRc[r, te] = est.predict_proba(Xc.iloc[te]), prior_of(yc[tr])
        mc = pd.DataFrame([ev.metrics(yc, Pc[r], PRc[r]) for r in range(R)]).mean().to_dict()
        cic = ev.bootstrap(yc, Pc.mean(0), PRc.mean(0), B=CFG["bootstrap"], seed=SEED)
        rows.append({"model": name, "method": "Complete-case analysis", "n": len(yc), **mc,
                     "macro_auc_lo": cic["macro_auc"][0], "macro_auc_hi": cic["macro_auc"][1]})
        si_ci = ev.bootstrap(zs["y"], zs["P"].mean(0), zs["PR"].mean(0), B=CFG["bootstrap"], seed=SEED)
        rows.append({"model": name, "method": "Single in-fold imputation (primary)", "n": len(y), **si,
                     "macro_auc_lo": si_ci["macro_auc"][0], "macro_auc_hi": si_ci["macro_auc"][1]})
        log(f"   {name}: MI {mm['macro_auc']:.3f}  single {si['macro_auc']:.3f}  CC {mc['macro_auc']:.3f} (n={len(yc)})")
    pd.DataFrame(rows).to_csv(RES / "e2mi.csv", index=False)


# ----------------------------------------------------------------------------- E9
def e9(df):
    """RUNBOOK R10: Duke-Breast-Cancer-MRI -- clinical vs DCE-MRI radiomics vs fusion (nested CV in Duke)."""
    log("E9 Duke MRI: clinical, radiomics and fusion models")
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    Xc, y, d = xy(df[df.cohort == "Duke MRI"], H.FEATURES_FULL)
    rad = H.duke_radiomics(ROOT / "data" / "raw").reindex(d.patient_id.to_numpy())
    rad = rad.apply(pd.to_numeric, errors="coerce")
    rad.columns = [f"rad_{i}" for i in range(rad.shape[1])]
    rad.index = Xc.index
    Xf = pd.concat([Xc, rad], axis=1)
    R, K = CFG["radiomics"]["repeats"], CFG["cv"]["outer_folds"]
    grid = {"clf__C": CFG["radiomics"]["C_grid"]}

    def pipe(cols_clin, cols_rad):
        tf = []
        if cols_clin:
            tf.append(("clin", Pipeline([("imp", M._imputer(SEED)), ("sc", StandardScaler())]), cols_clin))
        if cols_rad:
            tf.append(("rad", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols_rad))
        return Pipeline([("ct", ColumnTransformer(tf)), ("clf", LogisticRegression(max_iter=5000))])

    designs = {"clinical": (list(Xc.columns), []), "radiomics": ([], list(rad.columns)),
               "fusion": (list(Xc.columns), list(rad.columns))}
    out, rows = {}, []
    for lab, (cc, rc) in designs.items():
        P = np.zeros((R, len(y), 3))
        PR = np.zeros((R, len(y), 3))
        for r in range(R):
            for tr, te in StratifiedKFold(K, shuffle=True, random_state=SEED + r).split(Xf, y):
                gs = GridSearchCV(pipe(cc, rc), grid if rc else {"clf__C": [0.01, 0.1, 1, 10]},
                                  cv=StratifiedKFold(3, shuffle=True, random_state=SEED + r),
                                  scoring="neg_log_loss", n_jobs=4).fit(Xf.iloc[tr], y[tr])
                P[r, te], PR[r, te] = gs.predict_proba(Xf.iloc[te]), prior_of(y[tr])
        out[lab] = P.mean(0)
        mm = pd.DataFrame([ev.metrics(y, P[r], PR[r]) for r in range(R)]).mean().to_dict()
        ci = ev.bootstrap(y, P.mean(0), PR.mean(0), B=CFG["bootstrap"], seed=SEED)
        row = {"design": lab, "n": len(y), "n_features": len(cc) + len(rc), **mm,
               **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()}}
        if lab != "clinical":
            add_delta(row, y, P.mean(0), out["clinical"])
        rows.append(row)
        log(f"   {lab:9s} macroAUC={mm['macro_auc']:.3f}  G3-AUC={mm['g3_auc']:.3f}")
    pd.DataFrame(rows).to_csv(RES / "e9_duke_radiomics.csv", index=False)
    np.savez_compressed(RES / "e9_oof.npz", y=y, **out)


# ----------------------------------------------------------------------------- E10
def e10(df):
    """RUNBOOK R7: minimum sample size for the multinomial development model (Riley 2020 / Pate 2023)."""
    log("E10 sample-size calculation")
    z = np.load(RES / "e2_oof_mlr.npz")
    t, s_ = ev.multinomial_sample_size(z["y"], z["P"].mean(0), CFG["sample_size"]["n_params"],
                                       CFG["sample_size"]["shrinkage"], CFG["sample_size"]["delta"])
    t.to_csv(RES / "e10_sample_size_pairs.csv", index=False)
    s_.update({"n_params": CFG["sample_size"]["n_params"], "met": bool(s_["n_required"] <= s_["n_available"])})
    json.dump(s_, open(RES / "sample_size.json", "w"), indent=2)
    log(f"   required n = {s_['n_required']:.0f}; available {s_['n_available']}")


STAGES = {"e1": e1, "e2": e2, "e2s": e2s, "e3": e3, "e4": e4, "e5": e5, "e6": e6, "e7": e7, "e8": e8, "e2mi": e2mi, "e9": e9, "e2opt": e2opt,
          "e10": e10}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if which == ["all"]:
        which = [k for k in STAGES if k not in ("e2s", "e2opt")]
    df = load()
    t0 = time.time()
    for s in which:
        STAGES[s](df)
    json.dump({"stages": which, "seconds": round(time.time() - t0), "seed": SEED,
               "versions": {m: __import__(m).__version__ for m in
                            ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "shap", "scipy", "statsmodels",
                             "imblearn"]}},
              open(RES / f"run_{'_'.join(which)}.json", "w"), indent=2)
    log("done")

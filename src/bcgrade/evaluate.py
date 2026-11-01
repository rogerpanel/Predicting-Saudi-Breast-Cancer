"""Metrics, bootstrap CIs, calibration, decision-curve analysis, raking weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score, log_loss,
                             roc_auc_score)


def balanced_rule(P, prior):
    """Bayes decision under equal misclassification costs per class: argmax p_k / prior_k.
    Leaves probabilities (and thus calibration) untouched while preventing minority-class collapse."""
    prior = np.asarray(prior, float)
    if prior.ndim == 1:
        prior = prior[None, :]
    # tiny tie-break toward the more prevalent class, so a constant (prior) model predicts the majority class
    return np.argmax(P / prior + 1e-9 * prior, axis=1)


def _safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def cal_intercept_slope(y, p):
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:  # noqa: BLE001
        slope = inter = np.nan
    return inter, slope


def metrics(y, P, prior, weights=None, calib=True):
    """y in {0,1,2}; P (n,3) probabilities; prior = training class prevalence."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P)
    w = weights
    yhat = balanced_rule(P, prior)
    out = {}
    try:
        out["macro_auc"] = roc_auc_score(y, P, multi_class="ovr", average="macro", sample_weight=w,
                                         labels=[0, 1, 2])
    except ValueError:
        out["macro_auc"] = np.nan
    for k in range(3):
        yk = (y == k).astype(int)
        out[f"auc_g{k+1}"] = (roc_auc_score(yk, P[:, k], sample_weight=w) if 0 < yk.sum() < len(yk) else np.nan)
    out["macro_f1"] = f1_score(y, yhat, average="macro", sample_weight=w, labels=[0, 1, 2], zero_division=0)
    out["bal_acc"] = balanced_accuracy_score(y, yhat, sample_weight=w)
    out["qwk"] = cohen_kappa_score(y, yhat, weights="quadratic", sample_weight=w)
    onehot = np.eye(3)[y]
    out["brier"] = np.average(((P - onehot) ** 2).sum(1), weights=w)
    out["logloss"] = log_loss(y, np.clip(P, 1e-9, 1), labels=[0, 1, 2], sample_weight=w)
    y3 = (y == 2).astype(int)
    out["g3_auc"] = (roc_auc_score(y3, P[:, 2], sample_weight=w) if 0 < y3.sum() < len(y3) else np.nan)
    out["g3_brier"] = np.average((P[:, 2] - y3) ** 2, weights=w)
    if calib and w is None:
        out["g3_cal_int"], out["g3_cal_slope"] = cal_intercept_slope(y3, P[:, 2])
    return out


def fast_auc(yb, s):
    """Mann-Whitney AUC with mid-ranks for ties (equals sklearn roc_auc_score, unweighted)."""
    from scipy.stats import rankdata
    n1 = yb.sum()
    n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fast_metrics(y, P, prior):
    aucs = [fast_auc((y == k).astype(int), P[:, k]) for k in range(3)]
    yhat = balanced_rule(P, prior)
    f1 = []
    for k in range(3):
        tp = np.sum((yhat == k) & (y == k))
        fp = np.sum((yhat == k) & (y != k))
        fn = np.sum((yhat != k) & (y == k))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"macro_auc": np.mean(aucs), "g3_auc": aucs[2], "macro_f1": np.mean(f1),
            "brier": np.mean(((P - np.eye(3)[y]) ** 2).sum(1))}


def bootstrap(y, P, prior, keys=("macro_auc", "macro_f1", "g3_auc", "brier"), B=1000, seed=0):
    """Patient-level percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y)
    y, P = np.asarray(y).astype(int), np.asarray(P)
    prior = np.asarray(prior, float)
    res = {k: [] for k in keys}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 3:
            continue
        m = fast_metrics(y[i], P[i], prior[i] if prior.ndim == 2 else prior)
        for k in keys:
            res[k].append(m[k])
    return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) for k, v in res.items()}


def net_benefit(y, p, thresholds):
    y = np.asarray(y)
    n = len(y)
    rows = []
    prev = y.mean()
    for t in thresholds:
        pos = p >= t
        tp, fp = (pos & (y == 1)).sum(), (pos & (y == 0)).sum()
        rows.append({"threshold": t, "model": tp / n - fp / n * t / (1 - t),
                     "treat_all": prev - (1 - prev) * t / (1 - t), "treat_none": 0.0})
    return pd.DataFrame(rows)


def calibration_curve_lowess(y, p, frac=0.4):
    import statsmodels.api as sm
    o = np.argsort(p)
    fit = sm.nonparametric.lowess(np.asarray(y)[o], np.asarray(p)[o], frac=frac, return_sorted=True)
    return fit[:, 0], fit[:, 1]


def rake(df, targets: dict, max_iter=200, tol=1e-8):
    """Iterative proportional fitting. targets = {column: {level: share}}; rows with any NaN in
    the raking columns get weight 0. Returns weights normalised to mean 1 on included rows."""
    cols = list(targets)
    ok = df[cols].notna().all(axis=1).to_numpy()
    w = ok.astype(float)
    for _ in range(max_iter):
        prev = w.copy()
        for c, tgt in targets.items():
            v = df[c].to_numpy()
            tot = w[ok].sum()
            for lev, share in tgt.items():
                m = ok & (v == lev)
                cur = w[m].sum() / tot
                if cur > 0:
                    w[m] *= share / cur
        if np.abs(w - prev).max() < tol:
            break
    w[ok] /= w[ok].mean()
    return w


def ess(w):
    w = w[w > 0]
    return w.sum() ** 2 / (w ** 2).sum()


def paired_delta(y, PA, PB, B=1000, seed=0):
    """Patient-level paired bootstrap of AUC differences (A - B) on the same resamples.
    Returns {metric: (delta, lo, hi, two-sided bootstrap p)} for macro-AUC and G3-AUC."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    PA, PB = np.asarray(PA), np.asarray(PB)

    def both(i):
        a = [fast_auc((y[i] == k).astype(int), PA[i, k]) for k in range(3)]
        b = [fast_auc((y[i] == k).astype(int), PB[i, k]) for k in range(3)]
        return np.mean(a) - np.mean(b), a[2] - b[2]

    est = both(np.arange(len(y)))
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 3:
            bs.append(both(i))
    bs = np.array(bs)
    out = {}
    for j, k in enumerate(["macro_auc", "g3_auc"]):
        lo, hi = np.nanpercentile(bs[:, j], [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(bs[:, j] <= 0), np.mean(bs[:, j] >= 0)))
        out[k] = (est[j], lo, hi, p)
    return out


# ------------------------------------------------------------------ sample size (Riley 2020; Pate 2023)
def r2_cs(y, P, prior):
    """Cox-Snell R^2 from the log-likelihood of predictions P against the intercept-only model."""
    y = np.asarray(y).astype(int)
    n = len(y)
    ll = np.log(np.clip(P[np.arange(n), y], 1e-12, 1)).sum()
    ll0 = np.log(np.asarray(prior)[y]).sum()
    return 1 - np.exp(-2 * (ll - ll0) / n)


def max_r2_cs(prev):
    prev = np.asarray(prev, float)
    return 1 - np.exp(2 * np.sum(prev * np.log(prev)))


def multinomial_sample_size(y, P, n_params, shrinkage=0.9, delta=0.05, margin=0.05):
    """Minimum development sample size for a K-class multinomial model, following Riley et al.
    (BMJ 2020) criteria (i)-(iii) applied to each distinct pair of outcome categories as proposed by
    Pate et al. (SMMR 2023). P: optimism-free (out-of-fold) predictions giving the anticipated
    pairwise Cox-Snell R^2. n_params: predictor parameters per pairwise sub-model."""
    y = np.asarray(y).astype(int)
    K = P.shape[1]
    prev = np.bincount(y, minlength=K) / len(y)
    rows = []
    for k in range(K):
        for r in range(k + 1, K):
            m = (y == k) | (y == r)
            yy = (y[m] == k).astype(int)
            pk = P[m, k] / (P[m, k] + P[m, r])
            Pp = np.column_stack([1 - pk, pk])
            pp = np.array([1 - yy.mean(), yy.mean()])
            R2 = r2_cs(yy, Pp, pp)
            phi = prev[k] + prev[r]
            n1 = n_params / ((shrinkage - 1) * np.log(1 - R2 / shrinkage)) / phi
            mx = max_r2_cs(pp)
            S2 = R2 / (R2 + delta * mx)
            n2 = n_params / ((S2 - 1) * np.log(1 - R2 / S2)) / phi
            rows.append({"pair": f"G{k+1} vs G{r+1}", "prop_pair": phi, "r2cs": R2, "max_r2cs": mx,
                         "n_crit1": n1, "n_crit2": n2})
    n3 = max((1.96 / margin) ** 2 * p * (1 - p) for p in prev)
    t = pd.DataFrame(rows)
    return t, {"n_required": float(np.ceil(max(t.n_crit1.max(), t.n_crit2.max(), n3))),
               "n_crit3": float(np.ceil(n3)), "n_available": int(len(y)), "smallest_prev": float(prev.min())}


def dl_pool_logit(events, totals):
    """DerSimonian-Laird random-effects pooled proportion on the logit scale (with 95% CI, tau^2)."""
    e, n = np.asarray(events, float), np.asarray(totals, float)
    p = (e + 0.5) / (n + 1)
    th = np.log(p / (1 - p))
    v = 1 / (e + 0.5) + 1 / (n - e + 0.5)
    w = 1 / v
    fe = np.sum(w * th) / w.sum()
    Q = np.sum(w * (th - fe) ** 2)
    tau2 = max(0.0, (Q - (len(e) - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if len(e) > 1 else 0.0
    ws = 1 / (v + tau2)
    mu = np.sum(ws * th) / ws.sum()
    se = np.sqrt(1 / ws.sum())
    ex = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    return ex(mu), ex(mu - 1.96 * se), ex(mu + 1.96 * se), tau2

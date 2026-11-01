"""Scale-appropriate association statistics (replaces Pearson-on-category-codes).

nominal x any      -> bias-corrected Cramer's V (Bergsma 2013) with chi-square p
ordinal x ordinal  -> Kendall tau-b (binary variables are treated as ordinal)
continuous x ordinal -> Spearman rho (signed effect) + Kruskal-Wallis epsilon^2
All effects carry percentile-bootstrap 95% CIs; p-values are BH-adjusted across the family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cramers_v_bc(x, y):
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    n = tab.values.sum()
    r, k = tab.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return (np.sqrt(phi2 / denom) if denom > 0 else np.nan), p


def tau_b(x, y):
    t, p = stats.kendalltau(x, y, variant="b")
    return t, p


def spearman(x, y):
    r, p = stats.spearmanr(x, y)
    return r, p


def kw_epsilon2(cont, groups):
    gs = [cont[groups == g] for g in np.unique(groups)]
    gs = [g for g in gs if len(g) > 0]
    if len(gs) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*gs)
    n = len(cont)
    return h * (n + 1) / (n ** 2 - 1), p


MEASURES = {"V": cramers_v_bc, "tau_b": tau_b, "rho": spearman}


def assoc(df, a, b, kind, B=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    d = df[[a, b]].dropna()
    x, y = d[a].to_numpy(), d[b].to_numpy()
    f = MEASURES[kind]
    est, p = f(x, y)
    boots = []
    n = len(d)
    for _ in range(B):
        i = rng.integers(0, n, n)
        e, _ = f(x[i], y[i])
        boots.append(e)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    out = {"var_a": a, "var_b": b, "measure": kind, "n": n, "estimate": est, "ci_lo": lo, "ci_hi": hi, "p": p}
    if kind == "rho":
        out["kw_eps2"], out["kw_p"] = kw_epsilon2(x, y)
    return out


# pairs: (a, b, measure). grade is ordinal 1-3; hist nominal; binaries ordinal 0/1.
PAIRS = [
    ("grade", "er", "tau_b"), ("grade", "pr", "tau_b"), ("grade", "her2", "tau_b"),
    ("grade", "t_cat", "tau_b"), ("grade", "n_pos", "tau_b"), ("grade", "postmeno", "tau_b"),
    ("age", "grade", "rho"), ("grade", "histo", "V"),
    ("histo", "er", "V"), ("t_cat", "n_pos", "tau_b"), ("age", "er", "rho"),
]


def association_table(df, cohorts, B=1000, seed=20261017):
    rng = np.random.default_rng(seed)
    rows = []
    for c in cohorts:
        sub = df[df.cohort == c]
        for a, b, k in PAIRS:
            if sub[[a, b]].dropna().shape[0] < 30:
                continue
            r = assoc(sub, a, b, k, B=B, rng=rng)
            r["cohort"] = c
            rows.append(r)
    t = pd.DataFrame(rows)
    t["p_bh"] = multipletests(t.p, method="fdr_bh")[1]
    return t

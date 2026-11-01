"""Model ladder. Every learned step (imputation, scaling, resampling, model) lives inside one
Pipeline so it is re-fitted on training folds only -- the anti-leakage guarantee tested in
tests/test_leakage.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HIST_LEVELS = ["ilc", "mixed", "other"]  # IDC/NST is the reference level


def design(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Numeric design matrix; histology dummy-coded, missing histology -> NaN dummies (imputed in-fold)."""
    X = pd.DataFrame(index=df.index)
    for f in features:
        if f == "histo":
            for lev in HIST_LEVELS:
                X[f"hist_{lev}"] = np.where(df["histo"].isna(), np.nan, (df["histo"] == lev).astype(float))
        else:
            X[f] = df[f].astype(float)
    return X


class OrdinalLogit(BaseEstimator, ClassifierMixin):
    """Proportional-odds (cumulative logit) model with optional L2 penalty, fitted by L-BFGS."""

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    def fit(self, X, y):
        from scipy.optimize import minimize
        from scipy.special import expit
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        yi = np.searchsorted(self.classes_, y)
        p = X.shape[1]

        def unpack(w):
            beta, c0, d = w[:p], w[p], w[p + 1:]
            cuts = np.concatenate([[c0], c0 + np.cumsum(np.exp(d))])
            return beta, cuts

        def nll(w):
            beta, cuts = unpack(w)
            eta = X @ beta
            cdf = np.column_stack([np.zeros(len(X))] + [expit(c - eta) for c in cuts] + [np.ones(len(X))])
            pr = np.clip(cdf[np.arange(len(X)), yi + 1] - cdf[np.arange(len(X)), yi], 1e-12, 1)
            return -np.log(pr).sum() + self.alpha * beta @ beta

        w0 = np.zeros(p + K - 1)
        res = minimize(nll, w0, method="L-BFGS-B")
        self.coef_, self.cuts_ = unpack(res.x)
        return self

    def predict_proba(self, X):
        from scipy.special import expit
        eta = np.asarray(X, float) @ self.coef_
        cdf = np.column_stack([np.zeros(len(eta))] + [expit(c - eta) for c in self.cuts_] + [np.ones(len(eta))])
        return np.clip(np.diff(cdf, axis=1), 1e-9, 1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def to_float_array(X):
    """Writable float copy (works around a read-only-buffer issue in IterativeImputer with pandas>=3)."""
    return np.array(X, dtype=float, copy=True)


def _imputer(seed):
    return IterativeImputer(max_iter=10, random_state=seed, keep_empty_features=True)


def make_model(name: str, seed: int = 0, smote: bool = False):
    """Return (pipeline, param_grid). Labels must be 0/1/2."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    lib = {
        "majority": (DummyClassifier(strategy="prior"), {}),
        "mlr": (LogisticRegression(C=1.0, max_iter=2000), {"clf__C": [0.01, 0.1, 1, 10]}),
        "ordinal": (OrdinalLogit(), {}),
        "knn": (KNeighborsClassifier(), {"clf__n_neighbors": [15, 31, 61, 101], "clf__weights": ["uniform", "distance"]}),
        "rf": (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
               {"clf__min_samples_leaf": [5, 20, 50], "clf__max_features": [0.33, 0.66]}),
        "xgb": (XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8, random_state=seed,
                              n_jobs=1, verbosity=0, tree_method="hist", objective="multi:softprob"),
                {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.1]}),
        "lgbm": (LGBMClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                random_state=seed, n_jobs=1, verbose=-1),
                 {"clf__num_leaves": [4, 8, 16], "clf__min_child_samples": [20, 60]}),
    }
    if name == "tabpfn":  # optional tabular foundation model; needs gated weights (RUNBOOK R5)
        from tabpfn import TabPFNClassifier
        lib["tabpfn"] = (TabPFNClassifier(random_state=seed, device="cpu"), {})
    clf, grid = lib[name]
    from sklearn.preprocessing import FunctionTransformer
    steps = [("to_array", FunctionTransformer(to_float_array)), ("impute", _imputer(seed)), ("scale", StandardScaler())]
    if smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as IPipe
        return IPipe(steps + [("smote", SMOTE(random_state=seed)), ("clf", clone(clf))]), grid
    return Pipeline(steps + [("clf", clone(clf))]), grid


class MIClassifier(BaseEstimator, ClassifierMixin):
    """Multiple imputation with pooled predictions (RUNBOOK R4).

    fit(): m independent chained-equation imputations drawn from the posterior predictive
    (``sample_posterior=True``), each followed by its own scaler and a clone of ``estimator``,
    all fitted on the training data only. predict_proba(): the m probability matrices for new
    patients (each imputed by its own fitted imputer) are averaged ("pooled predictions")."""

    def __init__(self, estimator=None, m: int = 20, seed: int = 0):
        self.estimator, self.m, self.seed = estimator, m, seed

    def fit(self, X, y):
        from sklearn.preprocessing import FunctionTransformer
        self.pipes_ = []
        for i in range(self.m):
            imp = IterativeImputer(max_iter=10, random_state=self.seed + i, sample_posterior=True,
                                   keep_empty_features=True)
            pipe = Pipeline([("to_array", FunctionTransformer(to_float_array)), ("impute", imp),
                             ("scale", StandardScaler()), ("clf", clone(self.estimator))])
            self.pipes_.append(pipe.fit(X, y))
        self.classes_ = self.pipes_[0].classes_
        return self

    def predict_proba(self, X):
        import copy
        # copy so the posterior-sampling imputer's random state is not consumed: same X -> same output
        return np.mean([copy.deepcopy(p).predict_proba(X) for p in self.pipes_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def base_estimator(name: str, seed: int = 0, params: dict | None = None):
    """The bare classifier of ``make_model`` with tuned ``clf__`` parameters applied."""
    pipe, _ = make_model(name, seed)
    clf = clone(pipe.named_steps["clf"])
    if params:
        clf.set_params(**{k.replace("clf__", ""): v for k, v in params.items()})
    return clf


LADDER = ["majority", "mlr", "ordinal", "knn", "rf", "xgb", "lgbm"]
LABELS = {"majority": "Majority class", "mlr": "Multinomial LR", "ordinal": "Ordinal LR (prop. odds)",
          "knn": "k-NN", "rf": "Random forest", "xgb": "XGBoost", "lgbm": "LightGBM", "tabpfn": "TabPFN"}

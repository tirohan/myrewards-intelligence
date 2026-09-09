"""T-learner, X-learner, S-learner, and AIPW average treatment effect."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .overlap import assess_overlap


def _classifier(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _regressor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


@dataclass
class FittedUplift:
    """Scores from T-learner / X-learner / S-learner on the analysis frame."""

    mu0: np.ndarray
    mu1: np.ndarray
    tau: np.ndarray
    tau_t_learner: np.ndarray
    tau_x_learner: np.ndarray
    tau_s_learner: np.ndarray
    propensity: np.ndarray
    feature_names: list[str]
    row_attributions: list[str]
    mu0_model: Pipeline | None = None
    mu1_model: Pipeline | None = None
    tau0_model: Pipeline | None = None
    tau1_model: Pipeline | None = None
    s_model: Pipeline | None = None
    e_model: Pipeline | None = None


def _scaled_matrix(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(X))


def _tau_attributions(
    mu0_model: Pipeline,
    mu1_model: Pipeline,
    X: pd.DataFrame,
    *,
    k: int = 5,
) -> list[str]:
    """Linear τ attributions: (μ1 contrib − μ0 contrib) on the scaled features."""
    names = list(X.columns)
    s0 = _scaled_matrix(mu0_model, X)
    s1 = _scaled_matrix(mu1_model, X)
    c0 = mu0_model.named_steps["model"].coef_[0]
    c1 = mu1_model.named_steps["model"].coef_[0]
    rows: list[str] = []
    for i in range(len(X)):
        impacts = c1 * s1[i] - c0 * s0[i]
        contrib = [
            {
                "feature": names[j],
                "impact": float(impacts[j]),
                "direction": "increases" if impacts[j] >= 0 else "decreases",
            }
            for j in range(len(names))
        ]
        contrib.sort(key=lambda item: abs(item["impact"]), reverse=True)
        rows.append(json.dumps(contrib[:k]))
    return rows


def _predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        cls = int(model.named_steps["model"].classes_[0])
        return np.full(len(X), 1.0 if cls == 1 else 0.0)
    return proba[:, 1]


def scores_when_not_identified(
    n: int,
    feature_names: list[str],
    propensity: np.ndarray,
    treated_rate: float,
    *,
    ablation_mu0: np.ndarray | None = None,
    row_attributions: list[str] | None = None,
) -> FittedUplift:
    """Zero τ when CATE is not identified. μ0 prefers the ablation scorer."""
    zeros = np.zeros(n)
    mu0 = np.asarray(ablation_mu0, dtype=float) if ablation_mu0 is not None else zeros
    return FittedUplift(
        mu0=mu0,
        mu1=np.full(n, treated_rate if treated_rate == treated_rate else 0.0),
        tau=zeros,
        tau_t_learner=zeros,
        tau_x_learner=zeros,
        tau_s_learner=zeros,
        propensity=np.asarray(propensity, dtype=float),
        feature_names=feature_names,
        row_attributions=row_attributions or ["[]"] * n,
    )


def fit_uplift_models(
    df: pd.DataFrame,
    X: pd.DataFrame,
    *,
    random_state: int = 42,
) -> FittedUplift:
    """Fit T-learner and X-learner; τ is the X-learner score (imbalanced T)."""
    t = df["t_issued"].to_numpy().astype(int)
    y = df["y"].to_numpy().astype(int)
    overlap = assess_overlap(X, t, y)
    e = overlap.propensity
    if not overlap.cate_identified:
        return scores_when_not_identified(
            n=len(df),
            feature_names=list(X.columns),
            propensity=e,
            treated_rate=overlap.treated_closure_rate
            if overlap.treated_closure_rate == overlap.treated_closure_rate
            else 0.0,
        )

    treated = t == 1
    control = t == 0
    mu0_model = _classifier(random_state)
    mu1_model = _classifier(random_state)
    mu0_model.fit(X.loc[control], y[control])
    mu1_model.fit(X.loc[treated], y[treated])
    mu0 = _predict_proba(mu0_model, X)
    mu1 = _predict_proba(mu1_model, X)
    tau_t = mu1 - mu0

    d1 = y[treated] - mu0[treated]
    d0 = mu1[control] - y[control]
    tau1_model = _regressor()
    tau0_model = _regressor()
    tau1_model.fit(X.loc[treated], d1)
    tau0_model.fit(X.loc[control], d0)
    tau1 = tau1_model.predict(X)
    tau0 = tau0_model.predict(X)
    tau_x = e * tau0 + (1.0 - e) * tau1

    e_model = _classifier(random_state)
    e_model.fit(X, t)

    x_s = X.copy()
    x_s["__t"] = t
    s_model = _classifier(random_state)
    s_model.fit(x_s, y)
    x1 = X.copy()
    x0 = X.copy()
    x1["__t"] = 1
    x0["__t"] = 0
    tau_s = _predict_proba(s_model, x1) - _predict_proba(s_model, x0)

    return FittedUplift(
        mu0=mu0,
        mu1=mu1,
        tau=tau_x,
        tau_t_learner=tau_t,
        tau_x_learner=tau_x,
        tau_s_learner=tau_s,
        propensity=e,
        feature_names=list(X.columns),
        row_attributions=_tau_attributions(mu0_model, mu1_model, X),
        mu0_model=mu0_model,
        mu1_model=mu1_model,
        tau0_model=tau0_model,
        tau1_model=tau1_model,
        s_model=s_model,
        e_model=e_model,
    )


def apply_uplift_models(fitted: FittedUplift, X: pd.DataFrame) -> FittedUplift:
    """Score a new covariate frame with models trained on another split."""
    if fitted.mu0_model is None or fitted.mu1_model is None:
        n = len(X)
        zeros = np.zeros(n)
        return FittedUplift(
            mu0=zeros,
            mu1=zeros,
            tau=zeros,
            tau_t_learner=zeros,
            tau_x_learner=zeros,
            tau_s_learner=zeros,
            propensity=np.full(n, 0.5),
            feature_names=list(X.columns),
            row_attributions=["[]"] * n,
        )
    mu0 = _predict_proba(fitted.mu0_model, X)
    mu1 = _predict_proba(fitted.mu1_model, X)
    tau_t = mu1 - mu0
    if fitted.e_model is not None:
        e = np.clip(_predict_proba(fitted.e_model, X), 0.01, 0.99)
    else:
        e = np.clip(np.asarray(fitted.propensity[: len(X)], dtype=float), 0.01, 0.99)
    tau1 = fitted.tau1_model.predict(X) if fitted.tau1_model is not None else tau_t
    tau0 = fitted.tau0_model.predict(X) if fitted.tau0_model is not None else tau_t
    tau_x = e * tau0 + (1.0 - e) * tau1
    if fitted.s_model is not None:
        x1 = X.copy()
        x0 = X.copy()
        x1["__t"] = 1
        x0["__t"] = 0
        tau_s = _predict_proba(fitted.s_model, x1) - _predict_proba(fitted.s_model, x0)
    else:
        tau_s = tau_t
    return FittedUplift(
        mu0=mu0,
        mu1=mu1,
        tau=tau_x,
        tau_t_learner=tau_t,
        tau_x_learner=tau_x,
        tau_s_learner=tau_s,
        propensity=e,
        feature_names=list(X.columns),
        row_attributions=_tau_attributions(fitted.mu0_model, fitted.mu1_model, X),
        mu0_model=fitted.mu0_model,
        mu1_model=fitted.mu1_model,
        tau0_model=fitted.tau0_model,
        tau1_model=fitted.tau1_model,
        s_model=fitted.s_model,
        e_model=fitted.e_model,
    )


def aipw_ate(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    e: np.ndarray,
) -> dict[str, float]:
    """Doubly robust ATE with a normal-approx interval."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    e = np.clip(np.asarray(e, dtype=float), 0.01, 0.99)
    psi = (mu1 - mu0) + t * (y - mu1) / e - (1.0 - t) * (y - mu0) / (1.0 - e)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi))) if len(psi) > 1 else float("nan")
    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - 1.96 * se,
        "ci_high": ate + 1.96 * se,
        "n": float(len(psi)),
    }

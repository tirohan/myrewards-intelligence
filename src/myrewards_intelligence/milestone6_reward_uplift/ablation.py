"""Ablation P(close) without reward features.

Milestone 5 Model A with reward fields is leaky (AUC ~0.94). Targeting uses
this scorer instead — same allow-list as M6 covariates, trained in this
package so M6 never imports milestone5.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..core.config import resolve_path
from ..core.evidence import LabelSource

logger = logging.getLogger("myrewards_intelligence")


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


@dataclass
class AblationScorer:
    """Closure probabilities and linear attributions with reward fields excluded."""

    scores: np.ndarray
    feature_names: list[str]
    global_importance: list[dict[str, float | str]]
    row_attributions: list[str]
    model: Pipeline


def train_ablation_scorer(X: pd.DataFrame, y: np.ndarray) -> AblationScorer:
    """Fit logistic P(close | X) with no reward or leakage features."""
    y = np.asarray(y).astype(int)
    model = _pipeline()
    if len(np.unique(y)) < 2:
        n = len(X)
        return AblationScorer(
            scores=np.full(n, float(y.mean()) if n else 0.0),
            feature_names=list(X.columns),
            global_importance=[],
            row_attributions=["[]"] * n,
            model=model,
        )
    model.fit(X, y)
    scores = np.clip(model.predict_proba(X)[:, 1], 0.0, 1.0)
    coef = model.named_steps["model"].coef_[0]
    names = list(X.columns)
    importance = sorted(
        [
            {"feature": name, "weight": float(w), "abs_weight": float(abs(w))}
            for name, w in zip(names, coef, strict=True)
        ],
        key=lambda row: -row["abs_weight"],
    )
    transformed = model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(X))
    attributions: list[str] = []
    for row in transformed:
        contrib = [
            {
                "feature": names[i],
                "impact": float(coef[i] * row[i]),
                "direction": "increases" if coef[i] * row[i] >= 0 else "decreases",
            }
            for i in range(len(names))
        ]
        contrib.sort(key=lambda item: abs(item["impact"]), reverse=True)
        attributions.append(json.dumps(contrib[:5]))
    logger.info("Ablation scorer fit on %d rows / %d features", len(X), len(names))
    return AblationScorer(
        scores=scores,
        feature_names=names,
        global_importance=importance,
        row_attributions=attributions,
        model=model,
    )


def export_ablation_scores(
    df: pd.DataFrame,
    scorer: AblationScorer,
    path: str | Path,
) -> Path:
    """Write member × gap ablation scores for targeting (and M7)."""
    dest = resolve_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "member_id": df["member_id"].to_numpy(),
            "care_gap_code": df["care_gap_code"].to_numpy(),
            "ablation_score": scorer.scores,
            "label_source_basis": LabelSource.INCOMM_SOURCED.value,
            "excludes_reward_features": True,
            "model_version": "ablation_v1",
        }
    )
    out.to_csv(dest, index=False)
    logger.info("Wrote ablation scores: %s", dest)
    return dest


def load_ablation_scores(path: str | Path, df: pd.DataFrame) -> np.ndarray | None:
    """Join a previously exported ablation file onto the analysis frame."""
    dest = resolve_path(path)
    if not dest.exists():
        return None
    saved = pd.read_csv(dest)
    merged = df[["member_id", "care_gap_code"]].merge(
        saved[["member_id", "care_gap_code", "ablation_score"]],
        on=["member_id", "care_gap_code"],
        how="left",
    )
    if merged["ablation_score"].isna().all():
        return None
    return merged["ablation_score"].to_numpy()

"""Derived layer schema proposal for der_MemberImpactScores.

Per MILESTONE5_PLAN.md Section 6: modeled on the real, existing
der_MemberExperienceScores table structure.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings, resolve_path
from .features import prepare_features
from .train import TrainedModel

logger = logging.getLogger("myrewards_intelligence")

DDL_TEMPLATE = """-- der_MemberImpactScores DDL Proposal
-- Generated: {generated_date}
-- Model Version: {model_version}
-- Model Basis: {model_basis}
--
-- This is a PROPOSAL for InComm review. Do not apply to production without approval.
-- The table structure mirrors der_MemberExperienceScores for consistency.

CREATE TABLE [der_MemberImpactScores] (
    [Id] bigint NOT NULL IDENTITY,
    [MlPipelineRunId] bigint NOT NULL,
    [MemberId] nvarchar(100) NOT NULL,
    [CareGapCode] nvarchar(80) NOT NULL,
    [ScoredAtUtc] datetime2 NOT NULL,
    [ImpactScore] decimal(18,6) NOT NULL,      -- calibrated closure probability
    [ScoreTier] nvarchar(40) NOT NULL,          -- High / Medium / Low
    [ModelVersion] nvarchar(40) NOT NULL,
    [LabelSourceBasis] nvarchar(60) NOT NULL,   -- 'InComm-sourced' (Model A) or 'Expanded (mixed)' (Model B)
    [TopContributingFeaturesJson] nvarchar(max) NOT NULL,  -- top SHAP features
    CONSTRAINT [PK_der_MemberImpactScores] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_der_MemberImpactScores_der_MlPipelineRuns_MlPipelineRunId]
        FOREIGN KEY ([MlPipelineRunId]) REFERENCES [der_MlPipelineRuns] ([Id])
);

-- Index for member lookups
CREATE INDEX [IX_der_MemberImpactScores_MemberId]
    ON [der_MemberImpactScores] ([MemberId]);

-- Index for care gap queries
CREATE INDEX [IX_der_MemberImpactScores_CareGapCode]
    ON [der_MemberImpactScores] ([CareGapCode]);

-- Composite index for member + care gap lookups
CREATE INDEX [IX_der_MemberImpactScores_MemberId_CareGapCode]
    ON [der_MemberImpactScores] ([MemberId], [CareGapCode]);

-- Index for tier-based queries
CREATE INDEX [IX_der_MemberImpactScores_ScoreTier]
    ON [der_MemberImpactScores] ([ScoreTier]);

/*
COLUMN DOCUMENTATION:

Id: Auto-incrementing primary key.

MlPipelineRunId: Foreign key to der_MlPipelineRuns, tracking which pipeline run
    produced this score. Enables reproducibility and versioning.

MemberId: The member identifier, matching can_Members.MemberId.

CareGapCode: The care gap code this score applies to (e.g., DIAB_A1C_TEST).
    One member may have multiple rows for different care gaps.

ScoredAtUtc: Timestamp when the score was computed.

ImpactScore: The model's predicted probability of care gap closure (0.0 to 1.0).
    This is a calibrated probability, not just a raw model output.

ScoreTier: Categorical tier based on ImpactScore thresholds:
    - High: ImpactScore >= 0.7
    - Medium: 0.4 <= ImpactScore < 0.7
    - Low: ImpactScore < 0.4
    Thresholds are configurable and may be adjusted based on operational needs.

ModelVersion: Version identifier for the model that produced this score (e.g., "v1").

LabelSourceBasis: CRITICAL COLUMN - Indicates whether this score came from:
    - "InComm-sourced": Model A, trained only on InComm-confirmed closure labels.
      This is the primary, trustworthy model.
    - "Expanded (mixed)": Model B, trained on the full dataset including
      research-computed labels. Metrics should be interpreted with caution.

    This column ensures a reader can never mistake a Model-B-derived score
    for one trained purely on InComm's real data.

TopContributingFeaturesJson: JSON array of the top SHAP features explaining
    this individual score. Format:
    [
      {{"feature": "qualifying_claim_count", "impact": 0.23, "direction": "increases"}},
      {{"feature": "days_since_gap_opened", "impact": -0.15, "direction": "decreases"}}
    ]
    Mirrors the DomainScoresJson pattern in der_MemberExperienceScores.
*/
"""


def generate_ddl_proposal(
    trained_model: TrainedModel,
    output_path: str | Path | None = None,
) -> str:
    """Generate the der_MemberImpactScores DDL proposal.

    Args:
        trained_model: The trained model (for version and basis info)
        output_path: Optional path to write the DDL file

    Returns:
        The DDL SQL string
    """
    ddl = DDL_TEMPLATE.format(
        generated_date=datetime.now(UTC).isoformat(),
        model_version=trained_model.version,
        model_basis=trained_model.model_basis.value,
    )

    if output_path:
        path = resolve_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(ddl)
        logger.info("Wrote DDL proposal to: %s", path)

    return ddl


def compute_score_tier(score: float, high_threshold: float = 0.7, medium_threshold: float = 0.4) -> str:
    """Compute the score tier from an impact score."""
    if score >= high_threshold:
        return "High"
    elif score >= medium_threshold:
        return "Medium"
    else:
        return "Low"


def generate_scored_output(
    trained_model: TrainedModel,
    df: pd.DataFrame,
    shap_explanations: Any | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Generate scored output in der_MemberImpactScores format.

    This is a proposal output showing what the table would contain,
    not an actual database write.

    Args:
        trained_model: The trained model to use for scoring
        df: The dataset to score
        shap_explanations: Optional SHAP explanations for top features
        settings: Configuration settings

    Returns:
        DataFrame in der_MemberImpactScores format
    """
    if settings is None:
        from ..core.config import load_config
        settings = load_config()

    X, _ = prepare_features(df, fit_encoders=False, encoders=trained_model.encoders)

    # Pipeline/model handles NaN values internally
    scores = trained_model.model.predict_proba(X)[:, 1]

    score_tiers = settings.milestone5.score_tiers
    tiers = [
        compute_score_tier(s, score_tiers.high, score_tiers.medium)
        for s in scores
    ]

    scored_df = pd.DataFrame(
        {
            "MemberId": df["member_id"].values,
            "CareGapCode": df["care_gap_code"].values,
            "ScoredAtUtc": datetime.now(UTC).isoformat(),
            "ImpactScore": np.round(scores, 6),
            "ScoreTier": tiers,
            "ModelVersion": trained_model.version,
            "LabelSourceBasis": trained_model.model_basis.value,
            "TopContributingFeaturesJson": "[]",
        }
    )

    logger.info(
        "Generated scored output: %d rows, tier distribution: High=%d, Medium=%d, Low=%d",
        len(scored_df),
        (scored_df["ScoreTier"] == "High").sum(),
        (scored_df["ScoreTier"] == "Medium").sum(),
        (scored_df["ScoreTier"] == "Low").sum(),
    )

    return scored_df


def generate_member_prioritization(
    scored_df: pd.DataFrame,
    top_n_per_gap: int = 100,
) -> pd.DataFrame:
    """Generate member prioritization output ranked by impact score.

    Args:
        scored_df: The scored output DataFrame
        top_n_per_gap: Number of top members to include per care gap

    Returns:
        DataFrame with ranked prioritization
    """
    results = []

    for gap_code in sorted(scored_df["CareGapCode"].unique()):
        gap_df = scored_df[scored_df["CareGapCode"] == gap_code].copy()
        gap_df = gap_df.sort_values("ImpactScore", ascending=False)
        gap_df = gap_df.head(top_n_per_gap)
        gap_df["Rank"] = range(1, len(gap_df) + 1)
        results.append(gap_df)

    prioritized = pd.concat(results, ignore_index=True)

    return prioritized[
        ["Rank", "MemberId", "CareGapCode", "ImpactScore", "ScoreTier", "LabelSourceBasis"]
    ]

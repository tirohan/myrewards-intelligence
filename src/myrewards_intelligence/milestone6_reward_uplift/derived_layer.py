"""DDL proposal and scored dataframe for der_MemberRewardUpliftScores."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..core.evidence import LabelSource
from .domain_resolutions import OBSERVATION_WINDOW_DAYS
from .evidence import TreatmentSource
from .uplift import FittedUplift

DDL_TEMPLATE = """-- der_MemberRewardUpliftScores DDL Proposal
-- Generated: {generated_date}
-- THIS IS A PROPOSAL. Do not apply without InComm DBA review.
-- TreatmentSourceBasis is required so Track S scores cannot be read as Track R.

CREATE TABLE [der_MemberRewardUpliftScores] (
    [Id] bigint NOT NULL IDENTITY,
    [MlPipelineRunId] bigint NOT NULL,
    [MemberId] nvarchar(100) NOT NULL,
    [CareGapCode] nvarchar(80) NOT NULL,
    [ScoredAtUtc] datetime2 NOT NULL,
    [UpliftScore] decimal(18,6) NOT NULL,
    [UpliftTier] nvarchar(40) NOT NULL,
    [ControlOutcomeScore] decimal(18,6) NULL,
    [TreatedOutcomeScore] decimal(18,6) NULL,
    [PropensityToTreat] decimal(18,6) NULL,
    [ModelVersion] nvarchar(40) NOT NULL,
    [LabelSourceBasis] nvarchar(60) NOT NULL,
    [TreatmentSourceBasis] nvarchar(80) NOT NULL,
    [ObservationWindowDays] int NOT NULL,
    [TopContributingFeaturesJson] nvarchar(max) NOT NULL,
    CONSTRAINT [PK_der_MemberRewardUpliftScores] PRIMARY KEY ([Id]),
    CONSTRAINT [CK_der_MemberRewardUpliftScores_TreatmentSourceBasis]
        CHECK ([TreatmentSourceBasis] IN ({basis_values})),
    CONSTRAINT [CK_der_MemberRewardUpliftScores_Window] CHECK ([ObservationWindowDays] > 0),
    CONSTRAINT [FK_der_MemberRewardUpliftScores_der_MlPipelineRuns]
        FOREIGN KEY ([MlPipelineRunId]) REFERENCES [der_MlPipelineRuns] ([Id])
);

-- Runs are versioned by MlPipelineRunId; never overwrite a prior run's scores.
CREATE UNIQUE INDEX [UX_der_MemberRewardUpliftScores_Run_Member_Gap]
    ON [der_MemberRewardUpliftScores] ([MlPipelineRunId], [MemberId], [CareGapCode]);
CREATE INDEX [IX_der_MemberRewardUpliftScores_MemberId]
    ON [der_MemberRewardUpliftScores] ([MemberId]);
CREATE INDEX [IX_der_MemberRewardUpliftScores_CareGapCode]
    ON [der_MemberRewardUpliftScores] ([CareGapCode]);
CREATE INDEX [IX_der_MemberRewardUpliftScores_TreatmentSourceBasis]
    ON [der_MemberRewardUpliftScores] ([TreatmentSourceBasis]);
"""


def _tier(score: float) -> str:
    if score >= 0.10:
        return "High"
    if score >= 0.03:
        return "Medium"
    if score >= 0:
        return "Low"
    return "Suppress"


def generate_ddl_proposal() -> str:
    """Return the proposed SQL Server DDL."""
    basis_values = ", ".join(f"N'{t.value}'" for t in TreatmentSource)
    return DDL_TEMPLATE.format(generated_date=datetime.now(UTC).isoformat(), basis_values=basis_values)


def generate_scored_output(
    df: pd.DataFrame,
    fitted: FittedUplift,
    *,
    cate_identified: bool,
    treatment_source: str | None = None,
    observation_window_days: int | None = None,
) -> pd.DataFrame:
    """Build a der_MemberRewardUpliftScores-shaped dataframe. No live INSERT."""
    source = treatment_source or df.get(
        "treatment_source_basis", pd.Series([TreatmentSource.RESEARCH_SIMULATED.value] * len(df))
    )
    source_values = np.full(len(df), source) if isinstance(source, str) else source.to_numpy()

    tau = fitted.tau if cate_identified else np.zeros(len(df))
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    attributions = list(fitted.row_attributions) if fitted.row_attributions else ["[]"] * len(df)
    if len(attributions) != len(df):
        attributions = ["[]"] * len(df)
    return pd.DataFrame(
        {
            "MemberId": df["member_id"].to_numpy(),
            "CareGapCode": df["care_gap_code"].to_numpy(),
            "ScoredAtUtc": now,
            "UpliftScore": tau,
            "UpliftTier": [_tier(float(v)) for v in tau],
            "ControlOutcomeScore": fitted.mu0,
            "TreatedOutcomeScore": fitted.mu1,
            "PropensityToTreat": fitted.propensity,
            "ModelVersion": "v1",
            "LabelSourceBasis": LabelSource.INCOMM_SOURCED.value,
            "TreatmentSourceBasis": source_values,
            "ObservationWindowDays": observation_window_days or OBSERVATION_WINDOW_DAYS,
            "TopContributingFeaturesJson": attributions,
            "CateIdentified": cate_identified,
        }
    )

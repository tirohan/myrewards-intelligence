-- der_MemberImpactScores DDL Proposal
-- Generated: 2026-08-25T21:02:39.561529+00:00
-- Model Version: v1
-- Model Basis: InComm-sourced
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
      {"feature": "qualifying_claim_count", "impact": 0.23, "direction": "increases"},
      {"feature": "days_since_gap_opened", "impact": -0.15, "direction": "decreases"}
    ]
    Mirrors the DomainScoresJson pattern in der_MemberExperienceScores.
*/

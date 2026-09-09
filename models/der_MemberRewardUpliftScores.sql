-- der_MemberRewardUpliftScores DDL Proposal
-- Generated: 2026-09-09T11:38:38.512779+00:00
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
    CONSTRAINT [FK_der_MemberRewardUpliftScores_der_MlPipelineRuns]
        FOREIGN KEY ([MlPipelineRunId]) REFERENCES [der_MlPipelineRuns] ([Id])
);

CREATE INDEX [IX_der_MemberRewardUpliftScores_MemberId]
    ON [der_MemberRewardUpliftScores] ([MemberId]);
CREATE INDEX [IX_der_MemberRewardUpliftScores_CareGapCode]
    ON [der_MemberRewardUpliftScores] ([CareGapCode]);
CREATE INDEX [IX_der_MemberRewardUpliftScores_TreatmentSourceBasis]
    ON [der_MemberRewardUpliftScores] ([TreatmentSourceBasis]);

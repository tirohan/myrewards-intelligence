"""Treatment and assignment evidence tags for Milestone 6.

Track S (research-simulated) and Track R (InComm-issued) must never be mixed
in a headline metric.
"""

from __future__ import annotations

from enum import StrEnum


class TreatmentSource(StrEnum):
    """Source of the treatment indicator T (issued vs not)."""

    RESEARCH_SIMULATED = "Research-simulated"
    RESEARCH_GENERATED_LEDGER = "Research-generated IncentiveRewards mock"
    INCOMM_ISSUED = "InComm-issued"

    @property
    def description(self) -> str:
        if self == TreatmentSource.INCOMM_ISSUED:
            return (
                "Track R: dated issuance/redemption from a named InComm ledger. "
                "Milestone 7 may consume this as production-issued lift."
            )
        if self == TreatmentSource.RESEARCH_GENERATED_LEDGER:
            return (
                "Track R interface on a research-generated IncentiveRewards-shaped "
                "ledger (real members/gaps/amounts; assignment independent of Closed). "
                "Milestone 7 may consume this only as research-generated input, never as InComm-issued lift."
            )
        return (
            "Track S: treatment derived from sim_MemberRewardSimulations / "
            "HEALTH_ACTION_REWARD assignment. Method and pipeline only."
        )


def m7_use(source: TreatmentSource) -> str:
    """How Milestone 7 may use lift built on this treatment source."""
    return {
        TreatmentSource.INCOMM_ISSUED: "production",
        TreatmentSource.RESEARCH_GENERATED_LEDGER: "research-only",
        TreatmentSource.RESEARCH_SIMULATED: "never",  # Track S: T = Y, CATE not identified
    }[source]


class TreatmentGrain(StrEnum):
    """How T is joined onto (member, care gap) rows."""

    GAP_LEVEL = "gap-level"
    MEMBER_BROADCAST = "member-broadcast"


class AssignmentMechanism(StrEnum):
    """How members become treated, reconstructed from domain analysis."""

    ML_REWARD_ELIGIBLE = (
        "HEALTH_ACTION_REWARD recommendation cases from InComm ML "
        "(reward-eligible members); not a randomized holdout"
    )
    CAPACITY_MOCK = (
        "Research mock of IncentiveRewards capacity: Bernoulli issuance "
        "independent of Closed, lag 7-45 days after gap open"
    )
    UNKNOWN = "Unknown — ledger not named"

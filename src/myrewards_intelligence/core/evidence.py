"""Evidence source tracking — the structural home of the non-negotiable principle.

Every table, feature, and model output in this project distinguishes real
InComm-sourced data from research-generated/research-computed data, everywhere,
always. Never present a research-computed number as if InComm confirmed it.

This module defines the enums and sentinel values that enforce this principle.
"""

from __future__ import annotations

from enum import StrEnum


class LabelSource(StrEnum):
    """Source of care-gap closure labels.

    These values match the label_source column in the analytical dataset.
    """

    INCOMM_SOURCED = "InComm-sourced"
    RESEARCH_COMPUTED = "Research-computed (domain analysis)"
    RESEARCH_NO_EVIDENCE = "Research-computed (domain analysis) — no real evidence in current data"

    @classmethod
    def is_incomm_sourced(cls, value: str) -> bool:
        """Check if a label source value is InComm-sourced."""
        return value == cls.INCOMM_SOURCED.value

    @classmethod
    def values(cls) -> list[str]:
        """Return all possible label source values."""
        return [e.value for e in cls]


class FeatureSource(StrEnum):
    """Source of feature data."""

    INCOMM_SOURCED = "InComm-sourced"
    RESEARCH_GENERATED = "Research-generated (synthetic reward data)"
    DERIVED = "Derived (computed from InComm-sourced data)"

    @classmethod
    def values(cls) -> list[str]:
        """Return all possible feature source values."""
        return [e.value for e in cls]


class ModelBasis(StrEnum):
    """Label source basis for trained models.

    Model A uses only InComm-sourced labels and is the primary, trustworthy model.
    Model B uses the expanded dataset with mixed label sources.
    """

    MODEL_A_INCOMM = "InComm-sourced"
    MODEL_B_EXPANDED = "Expanded (mixed)"

    @property
    def description(self) -> str:
        """Human-readable description of this model basis."""
        if self == ModelBasis.MODEL_A_INCOMM:
            return (
                "Model A: Trained exclusively on InComm-sourced labels. "
                "This is the primary model whose metrics reflect real closure behavior."
            )
        return (
            "Model B: Trained on the expanded dataset including research-computed labels. "
            "Metrics must be interpreted with the label-source composition in mind."
        )


NOT_AVAILABLE = "NOT AVAILABLE — VALUE NOT YET PROVIDED"
"""Sentinel for values that require real InComm data not yet available.

Use this instead of inventing a number. The structure exists to hold the value;
the value is explicitly flagged as not yet available.
"""


NOT_COMPUTED = "NOT COMPUTED — DATABASE CONNECTION REQUIRED"
"""Sentinel for values that require a database connection to compute."""


def format_label_source_composition(
    incomm_count: int,
    research_count: int,
    total: int | None = None,
) -> str:
    """Format a human-readable label source composition string."""
    if total is None:
        total = incomm_count + research_count
    incomm_pct = (incomm_count / total * 100) if total > 0 else 0
    research_pct = (research_count / total * 100) if total > 0 else 0
    return (
        f"{incomm_count:,} InComm-sourced ({incomm_pct:.1f}%), "
        f"{research_count:,} research-computed ({research_pct:.1f}%)"
    )

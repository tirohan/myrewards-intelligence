"""Reward-amount elasticity or an explicit NOT_AVAILABLE."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.evidence import NOT_AVAILABLE
from .domain_resolutions import CATALOG_DEFAULT_AMOUNTS


def assess_elasticity(df: pd.DataFrame) -> dict[str, Any]:
    """Within-gap issued amount must vary; catalog defaults across gaps do not count."""
    if "issued_amount" not in df.columns:
        return {
            "status": NOT_AVAILABLE,
            "within_gap_amount_varies": False,
            "reason": "No issued_amount column.",
        }

    rows = []
    varies = False
    for gap, part in df.groupby("care_gap_code"):
        nunique = int(part["issued_amount"].nunique(dropna=True))
        rows.append(
            {
                "care_gap_code": gap,
                "n": int(len(part)),
                "distinct_amounts": nunique,
                "catalog_default": CATALOG_DEFAULT_AMOUNTS.get(str(gap)),
            }
        )
        if nunique > 1:
            varies = True

    if not varies:
        return {
            "status": NOT_AVAILABLE,
            "within_gap_amount_varies": False,
            "reason": (
                "Issued amount is the catalog default for each care gap. "
                "Cross-gap $15–$30 differences are confounded with gap type."
            ),
            "by_gap": rows,
        }

    return {
        "status": "ESTIMATED",
        "within_gap_amount_varies": True,
        "reason": "Within-gap issued amount varies; dose-response is in scope.",
        "by_gap": rows,
    }

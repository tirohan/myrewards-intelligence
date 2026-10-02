"""Reward-amount elasticity: dose-response of closure on offered amount, within care gap.

Estimator: linear probability model with care-gap fixed effects (Frisch-Waugh: demean y and amount within gap),
HC1 robust SE, on treated rows only. Identification needs amount to vary within a gap independently of outcome.
On the research mock it does by construction, so the result is a method + power statement, never InComm's
response curve. Where amount is constant within every gap the status stays NOT_AVAILABLE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.evidence import NOT_AVAILABLE
from .domain_resolutions import CATALOG_DEFAULT_AMOUNTS

MDE_Z = 2.8  # (1.96 + 0.84): 80% power, two-sided 5%
PER = 10.0  # report effects per $10


def _slope(y: np.ndarray, a: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on a (both already demeaned) with HC1 SE."""
    n = len(y)
    saa = float(a @ a)
    if n < 3 or saa == 0:
        return float("nan"), float("nan")
    beta = float(a @ y) / saa
    resid = y - beta * a
    var = float(np.sum((a * resid) ** 2)) / saa**2 * n / (n - 2)
    return beta, float(np.sqrt(var))


def assess_elasticity(df: pd.DataFrame) -> dict[str, Any]:
    """Within-gap amount variation decides whether dP(close)/d$ can be estimated at all."""
    if "issued_amount" not in df.columns:
        return {"status": NOT_AVAILABLE, "within_gap_amount_varies": False, "reason": "No issued_amount column."}

    offered = df[df["t_issued"].eq(1)] if "t_issued" in df.columns else df
    offered = offered[offered["issued_amount"].notna()]
    rows = []
    for gap, part in offered.groupby("care_gap_code"):
        rows.append(
            {
                "care_gap_code": gap,
                "n": int(len(part)),
                "distinct_amounts": int(part["issued_amount"].nunique()),
                "mean_amount": float(part["issued_amount"].mean()),
                "catalog_default": CATALOG_DEFAULT_AMOUNTS.get(str(gap)),
            }
        )
    varies = any(r["distinct_amounts"] > 1 for r in rows)
    if not varies:
        return {
            "status": NOT_AVAILABLE,
            "within_gap_amount_varies": False,
            "reason": (
                "Offered amount is constant within every care gap, so dose-response is not identified. "
                "Cross-gap differences are confounded with gap type."
            ),
            "by_gap": rows,
        }

    work = offered.assign(_y=offered["y"].astype(float), _a=offered["issued_amount"].astype(float))
    y_dm = work["_y"] - work.groupby("care_gap_code")["_y"].transform("mean")
    a_dm = work["_a"] - work.groupby("care_gap_code")["_a"].transform("mean")
    beta, se = _slope(y_dm.to_numpy(), a_dm.to_numpy())
    for r in rows:
        part = work[work["care_gap_code"] == r["care_gap_code"]]
        b, s = _slope(
            (part["_y"] - part["_y"].mean()).to_numpy(), (part["_a"] - part["_a"].mean()).to_numpy()
        )
        r["slope_pp_per_10usd"] = 100 * PER * b
        r["se_pp_per_10usd"] = 100 * PER * s
    return {
        "status": "ESTIMATED",
        "within_gap_amount_varies": True,
        "reason": (
            "Offered amount varies within gap (empirical InComm amount pool). Estimate is on the research mock, "
            "where amount is independent of outcome: it demonstrates the method and its power, not InComm's "
            "response curve."
        ),
        "pooled_slope_pp_per_10usd": 100 * PER * beta,
        "pooled_se_pp_per_10usd": 100 * PER * se,
        "minimum_detectable_slope_pp_per_10usd": 100 * PER * MDE_Z * se,
        "distinguishable_from_zero": bool(abs(beta) > 1.96 * se),
        "n_offered": int(len(work)),
        "by_gap": rows,
    }

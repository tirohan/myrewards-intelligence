"""Positivity (overlap) per stratum, with what to do when a stratum fails.

A stratum where every unit is treated has no observed untreated outcome, so tau is not identified there
and must never be extrapolated from other strata. We report:
  * Manski bounds on the average effect (no assumptions; and under monotone response: reward cannot lower closure)
  * the per-arm holdout size that would restore identification for a target effect
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

MIN_ARM_N = 50


def holdout_n_per_arm(p0: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two-proportion sample size per arm to detect an absolute lift `delta` over baseline `p0`."""
    p1 = min(p0 + delta, 0.999)
    pbar = (p0 + p1) / 2
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(np.ceil(num / delta**2))


def manski_bounds_all_treated(treated_rate: float) -> dict[str, float]:
    """ATE bounds when only treated outcomes are observed (outcome in [0, 1])."""
    return {
        "ate_lower_no_assumptions": treated_rate - 1.0,
        "ate_upper_no_assumptions": treated_rate,
        "ate_lower_monotone_response": 0.0,  # a reward cannot lower closure
        "ate_upper_monotone_response": treated_rate,
    }


def _row(name: str, part: pd.DataFrame, min_arm_n: int) -> dict[str, Any]:
    t = part["t_issued"].to_numpy()
    y = part["y"].to_numpy()
    n_t, n_c = int((t == 1).sum()), int((t == 0).sum())
    rate_t = float(y[t == 1].mean()) if n_t else float("nan")
    rate_c = float(y[t == 0].mean()) if n_c else float("nan")
    if n_c == 0:
        status = "NO_UNTREATED_ARM"
    elif n_t == 0:
        status = "NO_TREATED_ARM"
    elif min(n_t, n_c) < min_arm_n:
        status = "THIN_ARM"
    else:
        status = "OK"
    row: dict[str, Any] = {
        "stratum": name,
        "n": int(len(part)),
        "n_treated": n_t,
        "n_control": n_c,
        "treated_rate": rate_t,
        "control_rate": rate_c,
        "status": status,
    }
    if status != "OK":
        base = rate_c if n_c else (rate_t if n_t else 0.05)
        row["tau_reported"] = False
        row["holdout_n_per_arm_for_1pp"] = holdout_n_per_arm(max(base, 0.01), 0.01)
        row["holdout_n_per_arm_for_3pp"] = holdout_n_per_arm(max(base, 0.01), 0.03)
        if status == "NO_UNTREATED_ARM":
            row.update(manski_bounds_all_treated(rate_t))
    else:
        row["tau_reported"] = True
    return row


def positivity_table(df: pd.DataFrame, min_arm_n: int = MIN_ARM_N) -> list[dict[str, Any]]:
    """One row per condition family and per care gap."""
    work = df.assign(_family=df["care_gap_code"].astype(str).str.split("_").str[0])
    rows = [_row(f"family:{fam}", g, min_arm_n) for fam, g in work.groupby("_family")]
    rows += [_row(f"gap:{gap}", g, min_arm_n) for gap, g in work.groupby("care_gap_code")]
    return rows

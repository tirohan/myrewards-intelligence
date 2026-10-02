"""Per-member ROI, one-at-a-time (tornado) sensitivity, and Monte Carlo over lift and assumption uncertainty.

Everything is conditional on the Stars->QBP value model (stars.py: CMS public data + public prevalence anchors).
Probabilities here describe uncertainty about those inputs and the lift estimate; they are not forecasts of InComm
results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .assumptions import ELIGIBLE_SHARE, QBP_PER_ENROLLEE_YEAR
from .stars import StarValueModel


def member_roi(df: pd.DataFrame, model: StarValueModel, redeem_p: float, priced: bool) -> pd.DataFrame:
    """One row per (member, gap): expected cost, value of a closure, break-even tau, and net value if priced."""
    out = pd.DataFrame(
        {
            "MemberId": df["MemberId"].to_numpy(),
            "CareGapCode": df["CareGapCode"].to_numpy(),
            "UpliftScore": df["UpliftScore"].to_numpy(),
            "TreatmentSourceBasis": df["TreatmentSourceBasis"].to_numpy(),
        }
    )
    p_close = df["TreatedOutcomeScore"].clip(0, 1) if "TreatedOutcomeScore" in df.columns else pd.Series(1.0, index=df.index)
    out["expected_payout_cost_usd"] = (df["amount"] * p_close * redeem_p).to_numpy()
    for case in ("low", "base", "high"):
        out[f"closure_value_usd_{case}"] = [model.value(g, case) for g in df["CareGapCode"]]
    # lift this individual needs for the offer to pay for itself (base value); independent of measured lift
    with np.errstate(divide="ignore", invalid="ignore"):
        out["break_even_uplift_base"] = np.where(
            out["closure_value_usd_base"] > 0, out["expected_payout_cost_usd"] / out["closure_value_usd_base"], np.nan
        )
    for case in ("low", "base", "high"):
        net = out["UpliftScore"] * out[f"closure_value_usd_{case}"] - out["expected_payout_cost_usd"]
        out[f"expected_net_value_usd_{case}"] = net if priced else np.nan
    out["roi_status"] = "PRICED" if priced else "LIFT_NOT_DISTINGUISHABLE_FROM_ZERO"
    return out


def _mix_value(model: StarValueModel, counts: dict[str, int], **picks: str) -> float:
    n = sum(counts.values())
    return sum(c * model.components(g, **picks) for g, c in counts.items()) / n if n else float("nan")


def tornado(model: StarValueModel, counts: dict[str, int]) -> pd.DataFrame:
    """Swing in USD per closure (mean over the treated gap mix) when one input moves between its value-lowering and
    value-raising end and the rest stay at base. rho is the empirical boundary density; share is eligible share."""
    rows = []
    for name, key, prov in (
        ("qbp_per_enrollee_year", "qbp", QBP_PER_ENROLLEE_YEAR.provenance.value),
        ("rho_boundary_density (CMS 2026 data)", "rho", "Public data, computed (CMS 2026 Star Ratings)"),
        ("eligible_share (diabetes / hypertension)", "share", ELIGIBLE_SHARE["diabetes"].provenance.value),
    ):
        lo = _mix_value(model, counts, **{key: "low"})
        hi = _mix_value(model, counts, **{key: "high"})
        rows.append({"assumption": name, "provenance": prov, "value_at_low_input": lo,
                     "value_at_high_input": hi, "swing_usd": abs(hi - lo)})
    return pd.DataFrame(rows).sort_values("swing_usd", ascending=False).reset_index(drop=True)


def monte_carlo(
    ate_mean: float,
    n_treated: int,
    cost: float,
    ate_se: float,
    model: StarValueModel,
    counts: dict[str, int],
    draws: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Net value distribution: population lift ~ N(ate, se) per treated member; QBP, rho and eligible share each
    uniform between their value-lowering and value-raising ends (one draw per input, shared across gaps).

    Centred on the population ATE, NOT the in-sample uplift ranking gain of a top fraction: on noise, that fraction
    looks positive by selection (winner's curse) and the ranking has no validated gain here (AUUC ~ placebo).
    """
    rng = np.random.default_rng(seed)
    closures = n_treated * rng.normal(ate_mean, ate_se, draws)
    u_qbp, u_rho, u_share = rng.random(draws), rng.random(draws), rng.random(draws)
    qbp = QBP_PER_ENROLLEE_YEAR.low + u_qbp * (QBP_PER_ENROLLEE_YEAR.high - QBP_PER_ENROLLEE_YEAR.low)
    total = float(sum(counts.values()))
    value = np.zeros(draws)
    for gap, c in counts.items():
        if gap not in model.mapping.index or model.mapping.loc[gap, "cms_measure_id"] == "":
            continue
        r = model.mapping.loc[gap]
        d = model.density.loc[r["cms_measure_id"]]
        rho = d["rho_low"] + u_rho * (d["rho_high"] - d["rho_low"])
        sh = ELIGIBLE_SHARE[r["eligible_group"]]
        share = sh.high - u_share * (sh.high - sh.low)  # larger share lowers value
        pass_through = 1.0 if r["link"] == "direct" else float(d["mean_rate"])
        value += c / total * qbp * rho * pass_through / share / r["gaps_sharing_measure"]
    net = closures * value - cost
    return {
        "p_net_positive": float((net > 0).mean()),
        "net_p05_usd": float(np.percentile(net, 5)),
        "net_p50_usd": float(np.percentile(net, 50)),
        "net_p95_usd": float(np.percentile(net, 95)),
    }

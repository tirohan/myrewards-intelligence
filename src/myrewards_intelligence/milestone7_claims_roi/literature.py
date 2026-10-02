"""Priced ROI under a literature-anchored lift (an external prior, NOT InComm lift and NOT the mock lift).

Measured lift on the research mock is zero by construction and InComm has no randomized issuance data, so no
measured dollar ROI exists. The best-supported substitute is to price the program under effect sizes from published
incentive trials (assumptions.LITERATURE_RR: RR 1.00 / 1.17 / 1.42) and to report the relative risk the program needs
to break even. Both are scenarios, labeled as such, to be replaced by a pilot.

Effect is carried as an odds ratio (see assumptions.LITERATURE_OR) so P(close | offered) stays in [0, 1] for any
baseline. Per offer (pay on completion):
  p1      = OR x odds(baseline) / (1 + OR x odds(baseline))     P(close | offered)
  lift    = p1 - baseline;  implied RR = p1 / baseline
  cost    = amount x p1 x P(redeem)       paid only if the member closes, including would-have-closed-anyway
  benefit = lift x value of a closure     Stars->QBP range (stars.py), measure value split across its gaps
  break-even RR = V / (V - amount x P(redeem))   (closed form; undefined if V <= amount x P(redeem))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .assumptions import LITERATURE_OR, Provenance
from .stars import CASES, StarValueModel

BASIS = Provenance.LITERATURE.value
OR_CASES = {"low": LITERATURE_OR.low, "base": LITERATURE_OR.base, "high": LITERATURE_OR.high}


def literature_roi(
    df: pd.DataFrame,
    baselines: dict[str, float],
    star_model: StarValueModel,
    redeem_p: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-gap x lift scenario x value case, program-level offer-to-all grid)."""
    per_gap = df.groupby("CareGapCode").agg(n=("MemberId", "size"), amount=("amount", "mean")).reset_index()
    rows = []
    for r in per_gap.itertuples():
        base = baselines.get(r.CareGapCode)
        if base is None or not np.isfinite(base):
            continue
        odds0 = base / (1.0 - base)
        for rr_case, odds_ratio in OR_CASES.items():
            p1 = odds0 * odds_ratio / (1.0 + odds0 * odds_ratio)
            lift, rr = p1 - base, p1 / base
            cost = r.amount * p1 * redeem_p
            for v_case in CASES:
                v = star_model.value(r.CareGapCode, v_case)
                ar = r.amount * redeem_p
                rows.append(
                    {
                        "care_gap_code": r.CareGapCode,
                        "n_offers": int(r.n),
                        "baseline_closure": base,
                        "rr_case": rr_case,
                        "odds_ratio": odds_ratio,
                        "implied_rr": rr,
                        "p_close_if_offered": p1,
                        "lift_pp": 100 * lift,
                        "attributable_share_of_closers": lift / p1,  # 1 - 1/implied RR
                        "value_case": v_case,
                        "closure_value_usd": v,
                        "cost_per_offer_usd": cost,
                        "benefit_per_offer_usd": lift * v,
                        "net_per_offer_usd": lift * v - cost,
                        "roi": (lift * v - cost) / cost if cost > 0 else np.nan,
                        "break_even_rr": v / (v - ar) if v > ar else np.nan,
                        "break_even_feasible": bool(v > ar and base * v / (v - ar) < 1.0),
                        "stars_value_is_zero": bool(v == 0.0),
                        "basis": BASIS,
                    }
                )
    per = pd.DataFrame(rows)
    if per.empty:
        return per, per
    per["net_total_usd"] = per["net_per_offer_usd"] * per["n_offers"]
    per["cost_total_usd"] = per["cost_per_offer_usd"] * per["n_offers"]
    prog = (
        per.groupby(["rr_case", "odds_ratio", "value_case"])
        .agg(
            net_total_usd=("net_total_usd", "sum"),
            cost_total_usd=("cost_total_usd", "sum"),
            mean_implied_rr=("implied_rr", "mean"),
            attributable_share_of_closers=("attributable_share_of_closers", "mean"),
        )
        .reset_index()
    )
    prog["roi"] = prog["net_total_usd"] / prog["cost_total_usd"]
    prog["basis"] = BASIS
    return per, prog

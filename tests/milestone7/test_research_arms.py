import json

import numpy as np
import pandas as pd
import pytest

from myrewards_intelligence.core.evidence import NOT_AVAILABLE
from myrewards_intelligence.milestone7_claims_roi.assumptions import (
    LITERATURE_OR,
    LITERATURE_RR,
    odds_ratio_from_rr,
)
from myrewards_intelligence.milestone7_claims_roi.attribution import (
    load_real_timing,
    pn_bounds,
    pn_from_rr,
    timing_test,
)
from myrewards_intelligence.milestone7_claims_roi.literature import literature_roi
from myrewards_intelligence.milestone7_claims_roi.raf import raf_table
from tests.milestone7._model import make_model


def test_pn_bounds_and_attributable_fraction():
    b = pn_bounds(0.4, 0.2)
    assert b["pn_lower"] == pytest.approx(0.5) and b["pn_upper"] == 1.0 and b["pn_monotone"] == pytest.approx(0.5)
    harmed = pn_bounds(0.2, 0.4)  # an offer that lowers closure attributes nothing
    assert harmed["pn_lower"] == 0.0 and harmed["pn_monotone"] == 0.0
    assert pn_from_rr(1.25) == pytest.approx(0.2) and pn_from_rr(0.8) == 0.0


def test_real_timing_says_temporal_attribution_is_impossible(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"closed_gaps": 10, "claim_before_case": 10, "first_claim": "a", "last_claim": "b", "first_case": "c"}))
    t = load_real_timing(str(p))
    assert t["closures_after_an_offer"] == 0 and t["share_claims_before_case"] == 1.0
    assert load_real_timing(str(tmp_path / "missing.json"))["status"] == NOT_AVAILABLE


def _ledger(tmp_path, offered_faster: bool):
    rng = np.random.default_rng(0)
    n = 2000
    assigned = rng.random(n) < 0.5
    days = rng.integers(20, 300, n) - (60 * assigned if offered_faster else 0)
    t0 = pd.Timestamp("2025-01-01")
    pd.DataFrame(
        {
            "member_id": range(n), "care_gap_code": "G", "eligible": True, "assigned": assigned,
            "potential_issued_at": t0, "closed_at": t0 + pd.to_timedelta(np.maximum(days, 1), "D"),
        }
    ).to_csv(tmp_path / "e.csv", index=False)
    return str(tmp_path / "e.csv")


def test_timing_test_detects_acceleration_and_a_null(tmp_path):
    assert timing_test(_ledger(tmp_path, True), 365)["offer_accelerates_closure"] is True
    assert timing_test(_ledger(tmp_path, False), 365)["offer_accelerates_closure"] is False
    assert timing_test(str(tmp_path / "nope.csv"), 365)["status"] == NOT_AVAILABLE


def _dx(tmp_path, rows):
    p = tmp_path / "dx.csv"
    pd.DataFrame(rows, columns=["care_gap_code", "icd10_category", "closing_claims"]).to_csv(p, index=False)
    return str(p)


def test_raf_zero_for_labs_zero_for_nonpayment_dx_and_bounded_when_unverified(tmp_path):
    r = raf_table(_dx(tmp_path, [
        ("DIAB_A1C_TEST", "E11", 100),      # lab: not an RA-eligible encounter
        ("BH_POST_DISCHARGE_7D", "F41", 100),  # visit, but anxiety is not a payment HCC
        ("HTN_PCP_FOLLOWUP", "I50", 100),   # visit + payment HCC, factor unverified
    ])).set_index("care_gap_code")
    assert r.loc["DIAB_A1C_TEST", "raf_value_usd_base"] == 0.0
    assert r.loc["BH_POST_DISCHARGE_7D", "raf_value_usd_base"] == 0.0
    assert np.isnan(r.loc["HTN_PCP_FOLLOWUP", "raf_value_usd_base"])
    assert r.loc["HTN_PCP_FOLLOWUP", "raf_value_upper_bound_usd"] > 0
    assert r.loc["HTN_PCP_FOLLOWUP", "status"].startswith("NOT_AVAILABLE")


def test_raf_estimated_with_a_verified_factor(tmp_path):
    r = raf_table(_dx(tmp_path, [("CKD_NEPHROLOGY_VISIT", "E11", 100)])).iloc[0]
    # share 1.0 x P(not captured) 0.30 x V28 diabetes 0.166 x 12 x $1,100
    assert r["raf_value_usd_base"] == pytest.approx(0.30 * 0.166 * 12 * 1100)
    assert r["raf_value_usd_low"] < r["raf_value_usd_base"] < r["raf_value_usd_high"]


def _scores(n=100, amount=20.0):
    return pd.DataFrame({"MemberId": range(n), "CareGapCode": "DIAB_EYE_EXAM", "amount": amount})


def test_literature_effect_is_an_odds_ratio_and_never_exceeds_one():
    assert LITERATURE_OR.low == pytest.approx(1.0) and LITERATURE_OR.low < LITERATURE_OR.base < LITERATURE_OR.high
    assert odds_ratio_from_rr(LITERATURE_RR.high) > odds_ratio_from_rr(LITERATURE_RR.base)
    per, _ = literature_roi(_scores(), {"DIAB_EYE_EXAM": 0.95}, make_model(), 1.0)
    assert per["p_close_if_offered"].max() < 1.0  # a relative risk of 1.42 on 0.95 would have exceeded 1


def test_literature_no_effect_loses_the_payouts_and_break_even_is_closed_form():
    base, amount, redeem = 0.3, 20.0, 1.0
    model = make_model()
    per, prog = literature_roi(_scores(amount=amount), {"DIAB_EYE_EXAM": base}, model, redeem)
    none = per[(per.rr_case == "low") & (per.value_case == "base")].iloc[0]
    assert none["lift_pp"] == pytest.approx(0.0, abs=1e-9)
    assert none["net_per_offer_usd"] == pytest.approx(-none["cost_per_offer_usd"])  # windfall payouts only
    v = model.value("DIAB_EYE_EXAM", "base")
    row = per[(per.rr_case == "base") & (per.value_case == "base")].iloc[0]
    assert row["break_even_rr"] == pytest.approx(v / (v - amount * redeem))
    p1 = base * row["break_even_rr"]  # at the break-even RR, net is zero
    assert (p1 - base) * v - amount * p1 * redeem == pytest.approx(0.0, abs=1e-9)
    assert set(prog["rr_case"]) == {"low", "base", "high"}


def test_literature_marks_gaps_with_no_star_measure_as_zero_value():
    df = pd.DataFrame({"MemberId": range(10), "CareGapCode": "COPD_SPIROMETRY", "amount": 20.0})
    per, _ = literature_roi(df, {"COPD_SPIROMETRY": 0.5}, make_model(), 1.0)
    assert per["stars_value_is_zero"].all() and (per["benefit_per_offer_usd"] == 0).all()
    assert (per["net_per_offer_usd"] < 0).all()  # reward cost with no Stars value


def test_avoided_cost_arm_bh_only_zero_low_case_and_never_negative():
    from myrewards_intelligence.milestone7_claims_roi.avoided_cost import avoided_value, combine

    assert avoided_value("BH_POST_DISCHARGE_7D", "low") == pytest.approx(0.0, abs=1e-6)  # no causal effect in the low case
    lo, base, hi = (avoided_value("BH_POST_DISCHARGE_7D", c) for c in ("low", "base", "high"))
    assert 0.0 < base < hi and hi < 0.36 * 15_500  # cannot exceed the whole readmission cost
    assert np.isnan(avoided_value("DIAB_EYE_EXAM", "base"))
    grid = pd.DataFrame(
        {
            "care_gap_code": ["BH_POST_DISCHARGE_7D", "DIAB_EYE_EXAM"], "value_case": ["base", "base"],
            "closure_value_usd": [0.0, 100.0], "lift_pp": [5.0, 5.0], "cost_per_offer_usd": [10.0, 10.0],
            "p_close_if_offered": [0.8, 0.5],
        }
    )
    c = combine(grid)
    assert c.loc[0, "value_with_avoided_usd"] == pytest.approx(base)
    assert c.loc[1, "value_with_avoided_usd"] == 100.0 and "NOT_MODELLED" in c.loc[1, "avoided_cost_status"]
    assert c.loc[1, "break_even_rr_with_avoided"] == pytest.approx(100 / (100 - 20))  # unchanged gap: V/(V-amount)

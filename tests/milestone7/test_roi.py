import numpy as np
import pandas as pd
import pytest

from myrewards_intelligence.core.evidence import NOT_AVAILABLE
from myrewards_intelligence.milestone7_claims_roi.roi import (
    BASIS,
    lift_status,
    load_scores,
    scenarios,
)
from tests.milestone7._model import make_model

SIG = {"ate_mean": 0.02, "ate_se": 0.005}  # distinguishable from zero
NULL = {"ate_mean": -0.002, "ate_se": 0.005}


def _scores(tmp_path, basis=BASIS):
    p = tmp_path / "s.csv"
    pd.DataFrame(
        {
            "CareGapCode": ["DIAB_A1C_TEST", "HTN_PCP_FOLLOWUP", "CKD_NEPHROLOGY_VISIT"],
            "UpliftScore": [0.2, 0.1, 0.9],
            "CateIdentified": [True] * 3,
            "TreatmentSourceBasis": [basis] * 3,
        }
    ).to_csv(p, index=False)
    return str(p)


def test_excludes_ckd_and_prices_breakeven(tmp_path):
    r = scenarios(load_scores(_scores(tmp_path)), (1.0,), **SIG).iloc[0]
    assert r["n_treated"] == 2 and r["pricing_status"] == "PRICED"
    assert r["breakeven_value_per_closure_usd"] == pytest.approx(35 / 0.3)
    assert np.isnan(r["roi"]) and "RAF: separate arm" in r["financial_slots_status"]


def test_cost_uses_expected_redemption(tmp_path):
    r = scenarios(load_scores(_scores(tmp_path)), (1.0,), redeem_p=0.5, **SIG).iloc[0]
    assert r["reward_cost_issued_usd"] == 35 and r["reward_cost_expected_usd"] == 17.5
    assert r["cost_model"] == "pay_on_issuance_upper_bound"


def test_cost_is_paid_on_completion_not_on_offer():
    df = pd.DataFrame(
        {
            "CareGapCode": ["DIAB_A1C_TEST", "DIAB_A1C_TEST"],
            "UpliftScore": [0.1, 0.1],
            "TreatedOutcomeScore": [0.5, 0.0],  # second member never closes, so is never paid
            "amount": [20.0, 20.0],
        }
    )
    r = scenarios(df, (1.0,), redeem_p=1.0, **SIG).iloc[0]
    assert r["cost_model"] == "pay_on_completion"
    assert r["reward_cost_expected_usd"] == pytest.approx(10.0)  # 20 x 0.5; offer alone costs nothing


def test_refuses_to_price_noise(tmp_path):
    r = scenarios(load_scores(_scores(tmp_path)), (1.0,), value_per_closure=350, **NULL).iloc[0]
    assert r["pricing_status"] == "LIFT_NOT_DISTINGUISHABLE_FROM_ZERO"
    assert np.isnan(r["breakeven_value_per_closure_usd"]) and np.isnan(r["roi"])


def test_value_is_tagged_assumption(tmp_path):
    r = scenarios(load_scores(_scores(tmp_path)), (1.0,), value_per_closure=350, **SIG).iloc[0]
    assert r["roi"] == pytest.approx(2.0) and "Research assumption" in r["value_basis"]


def test_missing_uncertainty_is_not_priced():
    assert lift_status(np.nan, np.nan) == NOT_AVAILABLE


def test_rejects_non_mock(tmp_path):
    with pytest.raises(ValueError):
        load_scores(_scores(tmp_path, "InComm-issued"))


def test_required_lift_is_independent_of_measured_lift(tmp_path):
    r = scenarios(load_scores(_scores(tmp_path)), (1.0,), redeem_p=1.0, star_model=make_model(), **NULL).iloc[0]
    # cost 35 over 2 opportunities at base value v: required lift (pp) = 100 * 35 / (v * 2)
    v = r["stars_value_per_closure_usd_base"]
    assert r["required_lift_pp_for_breakeven_base"] == pytest.approx(100 * 35 / (v * 2))
    assert r["required_lift_pp_for_breakeven_low"] > r["required_lift_pp_for_breakeven_base"] > r["required_lift_pp_for_breakeven_high"]
    assert np.isnan(r["stars_roi_base"])  # lift not distinguishable, so no ROI


def test_stars_roi_only_when_priced(tmp_path):
    r = scenarios(load_scores(_scores(tmp_path)), (1.0,), redeem_p=1.0, star_model=make_model(), **SIG).iloc[0]
    assert r["pricing_status"] == "PRICED" and r["stars_roi_high"] > r["stars_roi_base"] > r["stars_roi_low"]

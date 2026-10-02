import numpy as np
import pandas as pd
import pytest

from myrewards_intelligence.core.evidence import NOT_AVAILABLE
from myrewards_intelligence.milestone7_claims_roi.assumptions import (
    ELIGIBLE_SHARE,
    QBP_PER_ENROLLEE_YEAR,
)
from myrewards_intelligence.milestone7_claims_roi.projections import (
    member_roi,
    monte_carlo,
    tornado,
)
from myrewards_intelligence.milestone7_claims_roi.roi import scenarios
from myrewards_intelligence.milestone7_claims_roi.validation import validate_closures
from tests.milestone7._model import make_model

MODEL = make_model()
COUNTS = {"DIAB_A1C_TEST": 50, "DIAB_BLOOD_SUGAR": 50, "HTN_PCP_FOLLOWUP": 50, "COPD_SPIROMETRY": 50}


def _df():
    return pd.DataFrame(
        {
            "MemberId": ["m1", "m2", "m3", "m4"],
            "CareGapCode": ["DIAB_A1C_TEST", "DIAB_BLOOD_SUGAR", "HTN_PCP_FOLLOWUP", "COPD_SPIROMETRY"],
            "UpliftScore": [0.1, 0.1, 0.1, 0.1],
            "TreatedOutcomeScore": [0.5, 0.5, 0.5, 0.5],
            "amount": [20.0, 20.0, 20.0, 20.0],
            "TreatmentSourceBasis": ["Research-generated IncentiveRewards mock"] * 4,
        }
    )


def test_value_formula_direct_rate_and_none():
    qbp, share_d = QBP_PER_ENROLLEE_YEAR.base, ELIGIBLE_SHARE["diabetes"].base
    # direct (eye exam): qbp x rho / share; the model has no sibling gap on C11
    assert MODEL.value("DIAB_EYE_EXAM") == pytest.approx(qbp * 0.10 / share_d)
    # rate pass-through (A1C test -> C12) at the measure's own rate, split across the two gaps sharing C12
    assert MODEL.value("DIAB_A1C_TEST") == pytest.approx(qbp * 0.48 * 0.85 / share_d / 2)
    assert MODEL.value("COPD_SPIROMETRY") == 0.0  # no 2026 Part C Star measure
    assert MODEL.value("NOT_A_GAP") == 0.0


def test_value_ends_are_ordered_and_share_is_inverted():
    low, base, high = (MODEL.value("DIAB_EYE_EXAM", c) for c in ("low", "base", "high"))
    assert low < base < high
    # a larger eligible share lowers value: the value-lowering end uses the HIGH share
    assert MODEL.components("DIAB_EYE_EXAM", share="low") == pytest.approx(
        QBP_PER_ENROLLEE_YEAR.base * 0.10 / ELIGIBLE_SHARE["diabetes"].high
    )


def test_member_roi_zero_value_gap_has_no_break_even_and_unpriced_net_is_nan():
    r = member_roi(_df(), MODEL, 1.0, priced=False).set_index("CareGapCode")
    assert r.loc["COPD_SPIROMETRY", "closure_value_usd_base"] == 0.0
    assert np.isnan(r.loc["COPD_SPIROMETRY", "break_even_uplift_base"])  # reward can never pay for itself via Stars
    assert r["expected_net_value_usd_base"].isna().all()
    assert r.loc["DIAB_A1C_TEST", "break_even_uplift_base"] == pytest.approx(10.0 / MODEL.value("DIAB_A1C_TEST"))


def test_member_roi_net_when_priced():
    r = member_roi(_df(), MODEL, 1.0, priced=True).set_index("CareGapCode")
    assert r.loc["HTN_PCP_FOLLOWUP", "expected_net_value_usd_base"] == pytest.approx(
        0.1 * MODEL.value("HTN_PCP_FOLLOWUP") - 10.0
    )
    assert r.loc["COPD_SPIROMETRY", "expected_net_value_usd_base"] == pytest.approx(-10.0)


def test_scenarios_use_the_model_and_include_zero_value_gaps():
    kw = {"redeem_p": 1.0, "ate_mean": 0.02, "ate_se": 0.005, "star_model": MODEL}
    r = scenarios(_df(), (1.0,), **kw).iloc[0]
    expected = np.mean([MODEL.value(g) for g in _df()["CareGapCode"]])
    assert r["stars_value_per_closure_usd_base"] == pytest.approx(expected)


def test_tornado_is_ranked_and_labels_provenance():
    t = tornado(MODEL, COUNTS)
    assert t["swing_usd"].is_monotonic_decreasing and (t["swing_usd"] >= 0).all()
    assert any("Public" in p for p in t["provenance"])


def test_monte_carlo_centres_on_population_ate_not_ranking_gain():
    zero = monte_carlo(0.0, 1000, 10000.0, 0.0001, MODEL, COUNTS)  # no lift, tiny uncertainty -> never pays back
    assert zero["p_net_positive"] == 0.0
    # 1000 treated at +20pp lift = 200 closures x mean value (~$100+) vs a $1 cost: always positive
    big = monte_carlo(0.20, 1000, 1.0, 0.0001, MODEL, COUNTS)
    assert big["p_net_positive"] > 0.95
    assert big["net_p05_usd"] <= big["net_p50_usd"] <= big["net_p95_usd"]


def test_validation_not_available_without_export_and_measured_with_it(tmp_path):
    assert validate_closures(str(tmp_path / "missing.csv"))["status"] == NOT_AVAILABLE
    p = tmp_path / "cv.csv"
    pd.DataFrame(
        {
            "care_gap_code": ["A", "B"],
            "closed_gaps": [10, 10],
            "claim_found": [10, 10],
            "same_member": [10, 10],
            "cpt_in_definition": [10, 9],
            "date_matches": [10, 10],
            "in_window": [10, 10],
            "gaps_sharing_a_claim": [0, 4],
        }
    ).to_csv(p, index=False)
    r = validate_closures(str(p))
    assert r["status"] == "MEASURED" and r["all_checks_pass"] is False
    assert r["rates"]["cpt_in_definition"] == pytest.approx(0.95) and r["gaps_sharing_a_claim"] == 4
    assert r["attribution_status"] == NOT_AVAILABLE and np.isfinite(r["rates"]["claim_found"])

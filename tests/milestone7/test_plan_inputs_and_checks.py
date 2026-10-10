import pandas as pd
import pytest

from myrewards_intelligence.milestone7_claims_roi.stars import StarValueModel
from myrewards_intelligence.milestone7_claims_roi.validation import FAULTS, fault_injection_test
from tests.milestone7._model import make_model


def test_plan_share_replaces_public_anchor_and_lowers_value():
    m = make_model()
    base = m.value("DIAB_EYE_EXAM")
    plan = StarValueModel(m.density.reset_index(), m.mapping.reset_index(), {"diabetes": 0.52})
    assert plan.value("DIAB_EYE_EXAM") == pytest.approx(base * 0.26 / 0.52)  # base anchor is 0.26
    assert plan.value("HTN_PCP_FOLLOWUP") == pytest.approx(m.value("HTN_PCP_FOLLOWUP"))  # other groups untouched


def test_share_sweep_scale_is_base_over_value():
    m = make_model()
    sw = m.share_sweep({"DIAB_EYE_EXAM": 1}, "diabetes", [0.26, 0.52])
    assert sw["break_even_scale"].tolist() == pytest.approx([1.0, 2.0])


def test_position_value_is_concentrated():
    m = make_model()
    m.position_density = pd.DataFrame({
        "measure_id": ["C11"] * 3, "position": ["all_contracts", "rated_3.5_measure_near_cut", "rated_4.0_or_more"],
        "n_contracts": [500, 40, 230], "rho_base": [0.1, 0.9, 0.0],
    })
    assert m.components("DIAB_EYE_EXAM", position="all_contracts") == pytest.approx(m.value("DIAB_EYE_EXAM"), rel=0.05)
    assert m.components("DIAB_EYE_EXAM", position="rated_3.5_measure_near_cut") > 5 * m.value("DIAB_EYE_EXAM")
    assert m.components("DIAB_EYE_EXAM", position="rated_4.0_or_more") == 0.0


def test_checker_flags_every_injected_fault_and_no_clean_row():
    t = fault_injection_test(n=1200, seed=3).set_index("fault")
    assert all(t.loc[f, "recall"] == 1.0 for f in FAULTS)
    assert t.iloc[-1]["detected"] == 0

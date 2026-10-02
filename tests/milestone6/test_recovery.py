"""Semi-synthetic recovery: AIPW must recover a known effect and a known null.

Fails if nuisance models are miscalibrated (e.g. class_weight="balanced" on a
rare outcome), which produced a spurious +3pp ATE on the zero-effect mock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from myrewards_intelligence.milestone6_reward_uplift.uplift import (
    aipw_ate,
    apply_uplift_models,
    fit_uplift_models,
)


def _simulate(true_tau: float, n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    t = rng.binomial(1, 0.45, n)
    p0 = 1 / (1 + np.exp(-(-3.8 + 0.5 * X["x1"])))
    y = rng.binomial(1, np.clip(p0 + true_tau * t, 0, 1))
    return X, pd.DataFrame({"t_issued": t, "y": y})


@pytest.mark.parametrize("true_tau", [0.0, 0.03])
def test_aipw_recovers_known_effect(true_tau):
    X, d = _simulate(true_tau)
    half = len(X) // 2
    fitted = fit_uplift_models(d.iloc[:half].reset_index(drop=True), X.iloc[:half].reset_index(drop=True))
    test_X = X.iloc[half:].reset_index(drop=True)
    s = apply_uplift_models(fitted, test_X)
    test = d.iloc[half:]
    r = aipw_ate(test["y"].to_numpy(), test["t_issued"].to_numpy(), s.mu0, s.mu1, s.propensity)
    assert r["ci_low"] <= true_tau <= r["ci_high"], r
    assert 0 < s.mu0.mean() < 0.1  # calibrated to the rare base rate, not inflated


def test_cate_ranking_recovers_known_heterogeneity():
    """At the real sample size (~11.7k) and ~2% base rate, a 0-6pp effect driven by x2 is ranked correctly.

    Favorable case (strong, smooth effect): shows the method works when signal exists, not that the real
    effect is detectable.
    """
    from scipy.stats import spearmanr

    n, rng = 11669, np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    t = rng.binomial(1, 0.45, n)
    tau = 0.03 + 0.03 * np.tanh(2 * X["x2"])
    p0 = 1 / (1 + np.exp(-(-3.8 + 0.5 * X["x1"])))
    d = pd.DataFrame({"t_issued": t, "y": rng.binomial(1, np.clip(p0 + tau * t, 0, 1))})
    h = n // 2
    fitted = fit_uplift_models(d.iloc[:h].reset_index(drop=True), X.iloc[:h].reset_index(drop=True))
    s = apply_uplift_models(fitted, X.iloc[h:].reset_index(drop=True))
    assert spearmanr(s.tau, tau.iloc[h:])[0] > 0.8


def test_time_aware_split_is_grouped_and_later_in_time():
    from myrewards_intelligence.milestone6_reward_uplift.evaluate_uplift import grouped_split

    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "member_id": rng.integers(0, 150, n).astype(str),
            "care_gap_code": "G",
            "potential_issued_at": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), "D"),
        }
    )
    train, test = grouped_split(df, test_size=0.2, random_state=0, split_mode="time_aware")
    assert not set(df.member_id.iloc[train]) & set(df.member_id.iloc[test])
    first = df.groupby("member_id")["potential_issued_at"].transform("min")
    assert first.iloc[test].min() >= first.iloc[train].max()


def test_elasticity_recovers_known_dose_response_and_null():
    from myrewards_intelligence.milestone6_reward_uplift.elasticity import assess_elasticity

    rng = np.random.default_rng(3)
    n = 20000
    gap = rng.choice(["A", "B"], n)
    amount = rng.choice([10.0, 20.0, 30.0], n)
    base = np.where(gap == "A", 0.15, 0.30)
    for slope_per_10, label in [(0.0, "null"), (0.04, "effect")]:
        y = rng.binomial(1, np.clip(base + slope_per_10 * (amount - 20) / 10, 0, 1))
        df = pd.DataFrame(
            {"care_gap_code": gap, "t_issued": 1, "issued_amount": amount, "y": y}
        )
        r = assess_elasticity(df)
        lo = r["pooled_slope_pp_per_10usd"] - 1.96 * r["pooled_se_pp_per_10usd"]
        hi = r["pooled_slope_pp_per_10usd"] + 1.96 * r["pooled_se_pp_per_10usd"]
        assert lo <= 100 * slope_per_10 <= hi, (label, r["pooled_slope_pp_per_10usd"])


def test_elasticity_not_available_when_amount_constant_within_gap():
    from myrewards_intelligence.core.evidence import NOT_AVAILABLE
    from myrewards_intelligence.milestone6_reward_uplift.elasticity import assess_elasticity

    df = pd.DataFrame(
        {"care_gap_code": ["A", "A", "B", "B"], "t_issued": 1, "issued_amount": [10.0, 10.0, 25.0, 25.0], "y": [0, 1, 0, 1]}
    )
    assert assess_elasticity(df)["status"] == NOT_AVAILABLE


def test_positivity_flags_all_treated_stratum_and_bounds_it():
    from myrewards_intelligence.milestone6_reward_uplift.positivity import positivity_table

    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "care_gap_code": ["DIAB_X"] * 300 + ["HTN_Y"] * 400,
            "t_issued": [1] * 300 + [1] * 200 + [0] * 200,
            "y": rng.binomial(1, 0.2, 700),
        }
    )
    rows = {r["stratum"]: r for r in positivity_table(df)}
    diab = rows["family:DIAB"]
    assert diab["status"] == "NO_UNTREATED_ARM" and diab["tau_reported"] is False
    assert diab["ate_lower_monotone_response"] == 0.0 and diab["ate_upper_monotone_response"] == diab["treated_rate"]
    assert diab["holdout_n_per_arm_for_1pp"] > diab["holdout_n_per_arm_for_3pp"] > 0
    assert rows["family:HTN"]["status"] == "OK"

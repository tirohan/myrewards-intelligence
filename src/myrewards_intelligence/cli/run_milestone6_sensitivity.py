"""Milestone 6 sensitivity: observation window x split mode, and the unsourced mock parameters.

Offer probability and offer lag have no InComm source (InComm cases share one creation timestamp), so we show
the conclusion does not depend on them. Writes reports/milestone6_sensitivity.json. Does not touch the main M6
outputs. Track R research-generated ledger only.
"""

from __future__ import annotations

import sys

from ..core.config import load_config, write_json
from ..milestone6_reward_uplift.data import headline_label_rows, load_analytical_dataset
from ..milestone6_reward_uplift.evaluate_uplift import evaluate_uplift_multi_seed
from ..milestone6_reward_uplift.evidence import TreatmentSource
from ..milestone6_reward_uplift.incentive_rewards_mock import generate_incentive_rewards_mock
from ..milestone6_reward_uplift.treatment import attach_treatment

MDE_Z = 2.8  # (1.96 + 0.84)
WINDOWS = (90, 180, 365)
OFFER_P = (0.25, 0.45, 0.75)
LAGS = ((0, 0), (7, 45), (45, 120))


def _evaluate(df, cfg, window, split, **mock_overrides):
    mock_cfg = cfg.model_copy(update=mock_overrides)
    mock = generate_incentive_rewards_mock(df, mock_cfg)
    treated = attach_treatment(
        df,
        treatment_source=TreatmentSource.RESEARCH_GENERATED_LEDGER,
        events=mock.events,
        eligibility=mock.eligibility,
        window_days=window,
    )
    r = evaluate_uplift_multi_seed(treated, list(cfg.diagnostic_seeds), cfg.test_size, split)
    ate, se = r.get("ate_mean"), r.get("ate_se")
    return {
        "window_days": window,
        "split_mode": split,
        "offer_probability": mock_cfg.mock_treat_probability,
        "lag_days": [mock_cfg.mock_issue_lag_days_min, mock_cfg.mock_issue_lag_days_max],
        "n_rows": r["n_rows"],
        "n_treated": r["n_treated"],
        "treated_rate": r["treated_closure_rate"],
        "control_rate": r["control_closure_rate"],
        "ate_mean": ate,
        "ate_se": se,
        "distinguishable": bool(se and abs(ate) > 1.96 * se),
        "mde_abs": MDE_Z * se if se else None,
        "relative_mde": MDE_Z * se / r["control_closure_rate"] if se and r["control_closure_rate"] else None,
    }


def main() -> int:
    settings = load_config()
    cfg = settings.milestone6
    df = headline_label_rows(load_analytical_dataset(settings))
    rows = [
        _evaluate(df, cfg, w, s)
        for w in WINDOWS
        for s in ("grouped_stratified", "time_aware")
    ]
    # unsourced parameters: sweep at the InComm-mandated 365-day window
    rows += [
        _evaluate(
            df, cfg, 365, "grouped_stratified",
            mock_treat_probability=p, mock_issue_lag_days_min=lo, mock_issue_lag_days_max=hi,
        )
        for p in OFFER_P
        for lo, hi in LAGS
    ]
    write_json(
        {"treatment_source_basis": TreatmentSource.RESEARCH_GENERATED_LEDGER.value, "rows": rows},
        "reports/milestone6_sensitivity.json",
    )
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "n_rows"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Secondary cohorts on research-computed labels (currently CKD_NEPHROLOGY_VISIT).

InComm's rules engine never produced CKD gap rows (ConditionCode 'CKD' vs 'CHRONIC_KIDNEY_DISEASE'), so these
labels are ours. They run through the same Track R pipeline but are reported SEPARATELY and never enter the
headline metrics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.config import Milestone6Config
from ..core.evidence import LabelSource
from .domain_resolutions import SECONDARY_RESEARCH_CARE_GAPS
from .elasticity import assess_elasticity
from .evaluate_uplift import evaluate_uplift_multi_seed
from .evidence import TreatmentSource
from .incentive_rewards_mock import generate_incentive_rewards_mock
from .positivity import positivity_table
from .treatment import attach_treatment


def run_secondary_cohort(raw: pd.DataFrame, cfg: Milestone6Config) -> dict[str, Any]:
    rows = raw[
        (raw["label_source"] != LabelSource.INCOMM_SOURCED.value)
        & raw["care_gap_code"].isin(SECONDARY_RESEARCH_CARE_GAPS)
    ].copy()
    base = {
        "headline": False,
        "label_source": LabelSource.RESEARCH_COMPUTED.value,
        "care_gaps": list(SECONDARY_RESEARCH_CARE_GAPS),
        "n_rows": int(len(rows)),
    }
    if rows.empty:
        return {**base, "status": "NO_ROWS"}
    mock = generate_incentive_rewards_mock(rows, cfg, include_gaps=SECONDARY_RESEARCH_CARE_GAPS)
    treated = attach_treatment(
        rows,
        treatment_source=TreatmentSource.RESEARCH_GENERATED_LEDGER,
        events=mock.events,
        eligibility=mock.eligibility,
        window_days=cfg.observation_window_days,
    )
    res = evaluate_uplift_multi_seed(
        treated, list(cfg.diagnostic_seeds), cfg.test_size, cfg.split_mode
    )
    keep = (
        "n_rows", "n_members", "n_treated", "n_control", "treated_closure_rate", "control_closure_rate",
        "cate_identified", "ate_mean", "ate_se", "auuc_mean",
    )
    return {
        **base,
        "status": "OK",
        **{k: res.get(k) for k in keep},
        "positivity": positivity_table(treated),
        "elasticity": assess_elasticity(treated),
        "treatment_source_basis": TreatmentSource.RESEARCH_GENERATED_LEDGER.value,
    }

"""A tiny StarValueModel for tests (same schema as config/cms_star_boundary_density.csv and care_gap_cms_measure.csv)."""

import pandas as pd

from myrewards_intelligence.milestone7_claims_roi.stars import StarValueModel


def make_model() -> StarValueModel:
    density = pd.DataFrame(
        {
            "measure_id": ["C11", "C12", "C14"],
            "weight": [1, 3, 3],
            "n_contracts": [500, 500, 500],
            "mean_rate": [0.78, 0.85, 0.79],
            "rho_low": [0.09, 0.45, 0.43],
            "rho_base": [0.10, 0.48, 0.48],
            "rho_high": [0.11, 0.50, 0.52],
        }
    )
    mapping = pd.DataFrame(
        {
            "care_gap_code": ["DIAB_EYE_EXAM", "DIAB_A1C_TEST", "DIAB_BLOOD_SUGAR", "HTN_PCP_FOLLOWUP", "COPD_SPIROMETRY"],
            "cms_measure_id": ["C11", "C12", "C12", "C14", "", ][:5],
            "link": ["direct", "rate", "rate", "rate", "none"],
            "eligible_group": ["diabetes", "diabetes", "diabetes", "hypertension", ""],
        }
    )
    return StarValueModel(density, mapping)

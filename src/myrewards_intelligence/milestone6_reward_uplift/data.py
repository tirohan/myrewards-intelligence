"""Load the analytical extract for Milestone 6 without importing Milestone 5.

Track R uses an IncentiveRewards-shaped ledger (research-generated mock when
the 3rd-party feed is absent). Same EVENT_COLUMNS schema either way.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..core.config import Settings, get_dataset_path, load_config, resolve_path
from ..core.evidence import NOT_AVAILABLE, LabelSource
from .domain_resolutions import CATALOG_DEFAULT_AMOUNTS, EXCLUDE_CARE_GAPS
from .evidence import TreatmentSource

logger = logging.getLogger("myrewards_intelligence")

EVENT_COLUMNS = [
    "member_id",
    "care_gap_code",
    "issued_at",
    "amount",
    "redeemed",
    "source",
]


class LedgerNotAvailableError(FileNotFoundError):
    """Track R issuance ledger is not named or not on disk."""


def load_analytical_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Load the Milestone 4 analytical CSV (local or reference repo)."""
    if settings is None:
        settings = load_config()
    path = get_dataset_path(settings)
    logger.info("Loading analytical dataset from: %s", path)
    return pd.read_csv(path)


def headline_label_rows(df: pd.DataFrame) -> pd.DataFrame:
    """InComm-sourced labels only; drop gaps with no native InComm evidence."""
    if "label_source" not in df.columns:
        raise ValueError("Dataset must contain label_source")
    out = df[df["label_source"] == LabelSource.INCOMM_SOURCED.value].copy()
    if "care_gap_code" in out.columns:
        out = out[~out["care_gap_code"].isin(EXCLUDE_CARE_GAPS)].copy()
    logger.info("Headline analysis set: %d rows / %d members", len(out), out["member_id"].nunique())
    return out


def reconstruct_assignment_events(df: pd.DataFrame) -> pd.DataFrame:
    """Gap-level Track S events from the live-validated InComm assignment rule.

    Care-gap QA on the connected extract: IsRewardEligible iff Status='Closed'
    (0 violations). HEALTH_ACTION_REWARD cases / sim rows are generated for
    those closed eligible gaps (CaseSubtype = care_gap_code). That is
    assignment, not issuance. IncentiveRewards (Track R) is not implemented.

    Redeemed is always False here: simulated 65% claim timing is not Y-safe.
    """
    catalog = set(CATALOG_DEFAULT_AMOUNTS) - set(EXCLUDE_CARE_GAPS)
    if "status" not in df.columns or "care_gap_code" not in df.columns:
        raise ValueError("Assignment reconstruction needs status and care_gap_code")
    closed = df[df["care_gap_code"].isin(catalog) & (df["status"] == "Closed")].copy()
    logger.info(
        "Reconstructed gap-level assignment events: n=%d (Closed catalog gaps only)",
        len(closed),
    )
    return pd.DataFrame(
        {
            "member_id": closed["member_id"].to_numpy(),
            "care_gap_code": closed["care_gap_code"].to_numpy(),
            "issued_at": pd.NaT,
            "amount": closed["care_gap_code"].map(CATALOG_DEFAULT_AMOUNTS).to_numpy(),
            "redeemed": False,
            "source": TreatmentSource.RESEARCH_SIMULATED.value,
        }
    )


def events_from_analytical_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Track S adapter: member-level sim aggregates, no IssuedAtUtc on the extract."""
    work = df.copy()
    if "rewards_issued_count" not in work.columns:
        raise ValueError("Analytical dataset missing rewards_issued_count")
    claimed = work["rewards_claimed_count"] if "rewards_claimed_count" in work.columns else 0
    work["_claimed"] = pd.Series(claimed, index=work.index).fillna(0)
    members = (
        work.groupby("member_id", as_index=False)
        .agg(issued=("rewards_issued_count", "max"), claimed=("_claimed", "max"))
    )
    treated = members[members["issued"] > 0]
    return pd.DataFrame(
        {
            "member_id": treated["member_id"].to_numpy(),
            "care_gap_code": pd.NA,
            "issued_at": pd.NaT,
            "amount": pd.NA,
            "redeemed": treated["claimed"].to_numpy() > 0,
            "source": TreatmentSource.RESEARCH_SIMULATED.value,
        }
    )


def load_issuance_ledger(path: str | Path) -> pd.DataFrame:
    """Track R adapter: dated gap-level events from a named export.

    Empty path is NOT_AVAILABLE. A generated mock must already be on disk
    (or created via generate_incentive_rewards_mock) before this loader runs.
    Missing source is tagged Research-generated IncentiveRewards mock, never
    InComm-issued.
    """
    if not path:
        raise LedgerNotAvailableError(
            f"Track R issuance ledger path is empty. {NOT_AVAILABLE} "
            "Name the production issuance/redemption table (or a CSV export) "
            "in config milestone6.issuance_ledger_path."
        )
    dest = Path(path)
    if not dest.is_absolute():
        dest = resolve_path(dest)
    if not dest.exists():
        raise LedgerNotAvailableError(
            f"Track R ledger not found at {dest}. {NOT_AVAILABLE} "
            f"Expected columns: {EVENT_COLUMNS}."
        )
    events = pd.read_csv(dest)
    missing = [col for col in EVENT_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(f"Issuance ledger missing columns {missing}; need {EVENT_COLUMNS}")
    events["source"] = events["source"].fillna(TreatmentSource.RESEARCH_GENERATED_LEDGER.value)
    if (events["source"] == TreatmentSource.INCOMM_ISSUED.value).any():
        logger.warning(
            "Ledger source is InComm-issued. Confirm this is a named production "
            "export before feeding Milestone 7."
        )
    return events[EVENT_COLUMNS].copy()


def load_eligibility_sidecar(ledger_path: str | Path) -> pd.DataFrame | None:
    """Load the eligibility/window sidecar written next to a Track R ledger."""
    dest = Path(ledger_path)
    if not dest.is_absolute():
        dest = resolve_path(dest)
    side = dest.with_name(dest.stem + "_eligibility.csv")
    if not side.exists():
        return None
    return pd.read_csv(side)


def load_track_r_bundle(
    settings: Settings, analytical: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, float | int | str]]:
    """Track R events + eligibility. Generates the IncentiveRewards mock when configured."""
    cfg = settings.milestone6
    path = cfg.issuance_ledger_path or "models/incentive_rewards_mock.csv"
    if cfg.generate_incentive_rewards_mock:
        from .incentive_rewards_mock import generate_incentive_rewards_mock, write_mock_ledger

        mock = generate_incentive_rewards_mock(analytical, cfg)
        write_mock_ledger(mock, path)
        return mock.events, mock.eligibility, mock.assumptions
    events = load_issuance_ledger(path)
    return events, load_eligibility_sidecar(path), {}


def load_treatment_events(settings: Settings, analytical: pd.DataFrame) -> pd.DataFrame:
    """Dispatch Track S vs Track R. Same downstream schema either way."""
    track = (settings.milestone6.track or "S").upper()
    if track == "R":
        events, _, _ = load_track_r_bundle(settings, analytical)
        return events
    grain = (settings.milestone6.treatment_grain or "gap_level").lower()
    if grain == "member_broadcast":
        return events_from_analytical_aggregates(analytical)
    return reconstruct_assignment_events(analytical)

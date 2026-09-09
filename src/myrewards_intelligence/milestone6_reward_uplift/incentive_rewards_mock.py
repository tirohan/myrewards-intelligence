"""Research-generated IncentiveRewards-shaped ledger.

Daniel/InComm (2026-07-20 Q4) directed the team to mock issuance when the
3rd-party feed is absent. This mock uses real members, care gaps, and catalog
amounts. Assignment is independent of Closed — unlike HEALTH_ACTION_REWARD.

TreatmentSourceBasis stays Research-generated IncentiveRewards mock.
Milestone 7 must not consume these scores as InComm-issued lift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import Milestone6Config, resolve_path
from .data import EVENT_COLUMNS
from .domain_resolutions import CATALOG_DEFAULT_AMOUNTS, EXCLUDE_CARE_GAPS
from .evidence import TreatmentSource

logger = logging.getLogger("myrewards_intelligence")

# Real-world defaults (disclosed research assumptions, not InComm telemetry):
# outreach lag after gap identification; capacity so not everyone is issued;
# redemption in the range reported for health-incentive programs.
DEFAULT_TREAT_P = 0.45
DEFAULT_LAG_MIN = 7
DEFAULT_LAG_MAX = 45
DEFAULT_REDEEM_P = 0.55
DEFAULT_MEAN_DAYS_TO_CLAIM = 14.0
PROGRAM_NAME = "MyRewards Health Action Incentive"
REWARD_TYPE = "CARE_GAP_COMPLETION"
CURRENCY = "USD"


@dataclass
class MockIncentiveRewards:
    """Dated gap-level events plus a potential-issue time for every row."""

    events: pd.DataFrame
    eligibility: pd.DataFrame
    assumptions: dict[str, float | int | str]


def reconstruct_gap_dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Gap-open and closing-claim dates from the analytical extract."""
    as_of = pd.to_datetime(df["as_of_date"])
    tenure = pd.to_numeric(df["days_since_gap_opened"], errors="coerce").fillna(180)
    gap_open = as_of - pd.to_timedelta(tenure, unit="D")
    close = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if "days_since_last_qualifying_claim" not in df.columns:
        return as_of, gap_open, close
    closed = df["status"].eq("Closed")
    days_claim = pd.to_numeric(df["days_since_last_qualifying_claim"], errors="coerce")
    ok = closed & days_claim.notna()
    close.loc[ok] = as_of.loc[ok] - pd.to_timedelta(days_claim.loc[ok], unit="D")
    return as_of, gap_open, close


def generate_incentive_rewards_mock(
    df: pd.DataFrame,
    cfg: Milestone6Config | None = None,
    *,
    random_state: int | None = None,
) -> MockIncentiveRewards:
    """Build a capacity-constrained, time-ordered mock ledger on real rows.

    Real-world rule: do not issue after the gap already closed. Among gaps
    still open at the potential issue date, assignment is Bernoulli and
    independent of whether the gap later closed in the extract.
    """
    cfg = cfg or Milestone6Config()
    seed = int(random_state if random_state is not None else cfg.random_state)
    treat_p = float(getattr(cfg, "mock_treat_probability", DEFAULT_TREAT_P))
    lag_min = int(getattr(cfg, "mock_issue_lag_days_min", DEFAULT_LAG_MIN))
    lag_max = int(getattr(cfg, "mock_issue_lag_days_max", DEFAULT_LAG_MAX))
    redeem_p = float(getattr(cfg, "mock_redeem_probability", DEFAULT_REDEEM_P))
    rng = np.random.default_rng(seed)

    catalog = set(CATALOG_DEFAULT_AMOUNTS) - set(EXCLUDE_CARE_GAPS)
    work = df[df["care_gap_code"].isin(catalog)].copy().reset_index(drop=True)
    empty_events = pd.DataFrame(columns=EVENT_COLUMNS)
    empty_elig = pd.DataFrame(
        columns=["member_id", "care_gap_code", "potential_issued_at", "closed_at", "eligible", "assigned"]
    )
    if work.empty:
        return MockIncentiveRewards(events=empty_events, eligibility=empty_elig, assumptions={})

    as_of, gap_open, close = reconstruct_gap_dates(work)
    lag = rng.integers(lag_min, lag_max + 1, size=len(work))
    t0 = gap_open + pd.to_timedelta(lag, unit="D")
    latest = as_of - pd.Timedelta(days=1)
    too_late = t0 > latest
    already_closed = close.notna() & (close <= t0)
    missing_close_date = work["status"].eq("Closed") & close.isna()
    eligible = (~too_late.to_numpy()) & (~already_closed.to_numpy())
    draw = rng.uniform(0, 1, len(work)) < treat_p
    assigned = eligible & draw
    redeemed = assigned & (rng.uniform(0, 1, len(work)) < redeem_p)

    eligibility = pd.DataFrame(
        {
            "member_id": work["member_id"].to_numpy(),
            "care_gap_code": work["care_gap_code"].to_numpy(),
            "potential_issued_at": t0.to_numpy(),
            "closed_at": close.to_numpy(),
            "eligible": eligible,
            "assigned": assigned,
        }
    )
    treated = work.loc[assigned].copy()
    events = pd.DataFrame(
        {
            "member_id": treated["member_id"].to_numpy(),
            "care_gap_code": treated["care_gap_code"].to_numpy(),
            "issued_at": t0.loc[assigned].to_numpy(),
            "amount": treated["care_gap_code"].map(CATALOG_DEFAULT_AMOUNTS).to_numpy(),
            "redeemed": redeemed[assigned],
            "source": TreatmentSource.RESEARCH_GENERATED_LEDGER.value,
        }
    )
    assumptions = {
        "treat_probability": treat_p,
        "issue_lag_days_min": lag_min,
        "issue_lag_days_max": lag_max,
        "redeem_probability": redeem_p,
        "mean_days_to_claim": DEFAULT_MEAN_DAYS_TO_CLAIM,
        "assignment": "Bernoulli among gaps still open at t0; independent of later Closed",
        "amounts": "InComm catalog DefaultRewardAmount by care gap",
        "program_name": PROGRAM_NAME,
        "n_events": int(len(events)),
        "n_eligible_rows": int(eligible.sum()),
        "n_ineligible_already_closed": int(already_closed.sum()),
        "n_ineligible_after_as_of": int(too_late.sum()),
        "n_closed_without_claim_date": int(missing_close_date.sum()),
        "seed": seed,
    }
    logger.info(
        "IncentiveRewards mock: events=%d eligible=%d ineligible=%d treat_p=%.2f seed=%d",
        len(events),
        int(eligible.sum()),
        int((~eligible).sum()),
        treat_p,
        seed,
    )
    return MockIncentiveRewards(events=events, eligibility=eligibility, assumptions=assumptions)


def write_mock_ledger(mock: MockIncentiveRewards, path: str | Path) -> Path:
    """Write a real-world-shaped ledger CSV plus an eligibility sidecar."""
    dest = resolve_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = mock.events.copy()
    if len(out):
        issued = pd.to_datetime(out["issued_at"])
        out.insert(
            0,
            "incentive_reward_id",
            [
                f"IR-{mid}-{gap}-{ts.strftime('%Y%m%d')}"
                for mid, gap, ts in zip(out["member_id"], out["care_gap_code"], issued, strict=True)
            ],
        )
        out["program_name"] = PROGRAM_NAME
        out["reward_type"] = REWARD_TYPE
        out["currency"] = CURRENCY
    out.to_csv(dest, index=False)
    side = dest.with_name(dest.stem + "_eligibility.csv")
    mock.eligibility.to_csv(side, index=False)
    logger.info("Wrote mock ledger %s and %s", dest, side)
    return dest


def observation_outcome(
    eligibility: pd.DataFrame,
    df: pd.DataFrame,
    *,
    window_days: int,
) -> pd.Series:
    """Y = InComm closing claim in (potential_issued_at, +window], else 0."""
    merged = df[["member_id", "care_gap_code"]].merge(
        eligibility[["member_id", "care_gap_code", "potential_issued_at", "closed_at"]],
        on=["member_id", "care_gap_code"],
        how="left",
    )
    t0 = pd.to_datetime(merged["potential_issued_at"])
    closed_at = pd.to_datetime(merged["closed_at"])
    end = t0 + pd.to_timedelta(window_days, unit="D")
    y = (closed_at.notna() & (closed_at > t0) & (closed_at <= end)).astype(int)
    return pd.Series(y.to_numpy(), index=df.index)

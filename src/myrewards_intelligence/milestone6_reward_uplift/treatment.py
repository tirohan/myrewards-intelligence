"""Build the treatment indicator on (member, care gap) rows."""

from __future__ import annotations

import logging

import pandas as pd

from .domain_resolutions import CATALOG_DEFAULT_AMOUNTS
from .evidence import AssignmentMechanism, TreatmentGrain, TreatmentSource

logger = logging.getLogger("myrewards_intelligence")


def _mechanism_for(source: TreatmentSource) -> str:
    if source == TreatmentSource.RESEARCH_SIMULATED:
        return AssignmentMechanism.ML_REWARD_ELIGIBLE.value
    if source == TreatmentSource.RESEARCH_GENERATED_LEDGER:
        return AssignmentMechanism.CAPACITY_MOCK.value
    return AssignmentMechanism.UNKNOWN.value


def _t_from_events(df: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.Series, pd.Series, TreatmentGrain]:
    """Join a dated ledger onto (member, gap) rows. Gap-level if care_gap_code is filled."""
    work = events.copy()
    if "redeemed" not in work.columns:
        work["redeemed"] = False
    work["redeemed"] = work["redeemed"].astype("boolean").fillna(False).astype(bool)
    gap_level = work["care_gap_code"].notna().any() if "care_gap_code" in work.columns else False
    if gap_level:
        issued = (
            work.groupby(["member_id", "care_gap_code"], dropna=False)
            .agg(evt_n=("member_id", "size"), evt_redeemed=("redeemed", "max"))
            .reset_index()
        )
        merged = df[["member_id", "care_gap_code"]].merge(issued, on=["member_id", "care_gap_code"], how="left")
        t_issued = pd.Series(merged["evt_n"].fillna(0).gt(0).astype(int).to_numpy(), index=df.index)
        t_redeemed = pd.Series(
            merged["evt_redeemed"].astype("boolean").fillna(False).astype(bool).astype(int).to_numpy(),
            index=df.index,
        )
        return t_issued, t_redeemed, TreatmentGrain.GAP_LEVEL
    members = work.groupby("member_id").agg(
        evt_n=("member_id", "size"),
        evt_redeemed=("redeemed", "max"),
    )
    t_issued = df["member_id"].map(members["evt_n"]).fillna(0).gt(0).astype(int)
    t_redeemed = (
        df["member_id"].map(members["evt_redeemed"]).astype("boolean").fillna(False).astype(bool).astype(int)
    )
    return t_issued, t_redeemed, TreatmentGrain.MEMBER_BROADCAST


def attach_treatment(
    df: pd.DataFrame,
    *,
    treatment_source: TreatmentSource = TreatmentSource.RESEARCH_SIMULATED,
    events: pd.DataFrame | None = None,
    eligibility: pd.DataFrame | None = None,
    window_days: int | None = None,
) -> pd.DataFrame:
    """Attach T, Y, and evidence tags.

    Track S: sim rewards were rolled up to MemberId, so T is broadcast to every
    gap for that member (TreatmentGrain=member-broadcast) unless gap-level
    events are passed.
    Track R mock: pass events plus eligibility and window_days. Refuses to
    use status==Closed as Y when TreatmentSource is the research-generated ledger.
    """
    out = df.copy()
    if "status" not in out.columns:
        raise ValueError("Dataset must contain status")

    out["y"] = (out["status"] == "Closed").astype(int)
    if events is not None and len(events) and "source" in events.columns:
        raw = str(events["source"].iloc[0])
        allowed = {item.value for item in TreatmentSource}
        if raw in allowed:
            incoming = TreatmentSource(raw)
            if (
                incoming == TreatmentSource.INCOMM_ISSUED
                and treatment_source == TreatmentSource.RESEARCH_GENERATED_LEDGER
            ):
                logger.warning("Ignoring InComm-issued tag on a research-generated ledger")
            else:
                treatment_source = incoming
    if treatment_source == TreatmentSource.RESEARCH_GENERATED_LEDGER and (
        eligibility is None or not len(eligibility) or window_days is None
    ):
        raise ValueError(
            "Track R mock requires eligibility timestamps and window_days; "
            "refusing to fall back to status==Closed as Y."
        )
    if eligibility is not None and len(eligibility):
        keep_cols = [
            col
            for col in ("member_id", "care_gap_code", "eligible", "potential_issued_at", "closed_at")
            if col in eligibility.columns
        ]
        out = out.merge(eligibility[keep_cols], on=["member_id", "care_gap_code"], how="left")
        if "eligible" in out.columns:
            out = out.loc[out["eligible"].astype("boolean").fillna(False).astype(bool)].copy()
            out = out.reset_index(drop=True)
        if window_days is not None:
            from .incentive_rewards_mock import observation_outcome

            out["y"] = observation_outcome(eligibility, out, window_days=window_days).to_numpy()
    if events is not None:
        t_issued, t_redeemed, grain = _t_from_events(out, events)
        out["t_issued"] = t_issued.to_numpy()
        out["t_redeemed"] = t_redeemed.to_numpy()
        out["treatment_grain"] = grain.value
    else:
        if "rewards_issued_count" not in out.columns:
            raise ValueError("Dataset must contain rewards_issued_count")
        out["t_issued"] = (out["rewards_issued_count"].fillna(0) > 0).astype(int)
        claimed = out["rewards_claimed_count"] if "rewards_claimed_count" in out.columns else 0
        out["t_redeemed"] = (pd.Series(claimed, index=out.index).fillna(0) > 0).astype(int)
        out["treatment_grain"] = TreatmentGrain.MEMBER_BROADCAST.value
    out["issued_amount"] = out["care_gap_code"].map(CATALOG_DEFAULT_AMOUNTS)
    if events is not None and "amount" in events.columns and len(events):
        # gap-level offered amount (NaN for controls); falls back to the catalog default otherwise
        amt = (
            events.groupby(["member_id", "care_gap_code"], as_index=False)["amount"]
            .mean()
            .rename(columns={"amount": "_event_amount"})
        )
        out = out.merge(amt, on=["member_id", "care_gap_code"], how="left")
        out["issued_amount"] = out["_event_amount"].where(out["t_issued"].eq(1), out["issued_amount"])
        out = out.drop(columns="_event_amount")
    out["treatment_source_basis"] = treatment_source.value
    out["assignment_mechanism"] = _mechanism_for(treatment_source)

    issued_col = "issued_at" if "issued_at" in out.columns else "potential_issued_at"
    closed_col = "closed_at" if "closed_at" in out.columns else None
    if issued_col in out.columns and closed_col:
        issued = pd.to_datetime(out[issued_col], errors="coerce")
        closed = pd.to_datetime(out[closed_col], errors="coerce")
        treated_flag = out["t_issued"].eq(1) if "t_issued" in out.columns else False
        reverse = treated_flag & issued.notna() & closed.notna() & (issued > closed)
        dropped = int(reverse.sum())
        if dropped:
            logger.warning("Dropping %d rows with issuance after closure", dropped)
            out = out.loc[~reverse].copy()

    logger.info(
        "Treatment attached: n=%d treated=%d control=%d source=%s grain=%s",
        len(out),
        int(out["t_issued"].sum()),
        int((out["t_issued"] == 0).sum()),
        treatment_source.value,
        out["treatment_grain"].iloc[0] if len(out) else "",
    )
    return out

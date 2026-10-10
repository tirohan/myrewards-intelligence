"""Value of one incremental closure through Star Ratings -> QBP, from CMS's PUBLIC 2026 Star Ratings data.

InComm cannot share data, so nothing here is assumed from InComm. The boundary probability is measured on the real
published contracts (scripts/build_star_boundary_density.py):

  value of a closure = QBP$/enrollee x rho_m x pass_through / eligible_share_m / gaps_sharing_the_measure
    rho_m          P(one measure star step tips a 3.5-star contract to 4.0, the QBP) per unit of measure rate,
                   empirical over ~520 MA contracts using CMS cut points and weights (enrollment cancels)
    pass_through   1 for gaps that close the measure numerator (eye exam, KED); the measure's own rate for gaps that
                   are a precondition of it (HbA1c test -> Blood Sugar Controlled; PCP visit -> Controlling BP)
    eligible_share share of enrollees in the measure's denominator (public Medicare prevalence anchors, or a plan's own
                   share via plan_shares, e.g. {"diabetes": 0.40} for an SNP-heavy plan)
    gaps_sharing   one claim can close several gaps that roll up to the same measure, so the value is split

Gaps with no 2026 Part C Star measure (COPD spirometry, BH follow-up, CKD visit) are worth exactly 0 here.
low / base / high are the value-lowering / central / value-raising ends of each input, never mixed across classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .assumptions import ELIGIBLE_SHARE, QBP_PER_ENROLLEE_YEAR, Assumption, Provenance

CASES = ("low", "base", "high")
_SHARE_END = {"low": "high", "base": "base", "high": "low"}  # a LARGER eligible share lowers value


def _end(a: Assumption, pick: str) -> float:
    return getattr(a, pick)


@dataclass
class StarValueModel:
    density: pd.DataFrame  # measure_id, weight, n_contracts, mean_rate, rho_low, rho_base, rho_high
    mapping: pd.DataFrame  # care_gap_code, cms_measure_id, link, eligible_group
    plan_shares: dict[str, float] | None = None  # plan-supplied eligible share per group; replaces low/base/high
    position_density: pd.DataFrame | None = None  # measure_id, position, n_contracts, rho_base

    @classmethod
    def from_files(
        cls, density_path: str, mapping_path: str, position_path: str | None = None,
        plan_shares: dict[str, float] | None = None,
    ) -> StarValueModel:
        pos = pd.read_csv(position_path) if position_path and Path(position_path).exists() else None
        return cls(pd.read_csv(density_path), pd.read_csv(mapping_path), plan_shares or None, pos)

    def __post_init__(self) -> None:
        self.density = self.density.set_index("measure_id")
        m = self.mapping.fillna({"cms_measure_id": "", "eligible_group": ""}).set_index("care_gap_code")
        sharing = m[m["cms_measure_id"] != ""].groupby("cms_measure_id")["link"].transform("size")
        m["gaps_sharing_measure"] = sharing.reindex(m.index).fillna(1).astype(float)
        self.mapping = m

    def _share(self, group: str, share: str) -> float:
        if self.plan_shares and group in self.plan_shares:
            return float(self.plan_shares[group])
        return _end(ELIGIBLE_SHARE[group], _SHARE_END[share])

    def components(
        self, gap: str, qbp: str = "base", rho: str = "base", share: str = "base", position: str | None = None
    ) -> float:
        """USD per incremental closure of this gap with each input at the named end.

        position=None is the average over all rated contracts (a portfolio view). A named position (see
        position_density) conditions on where the contract sits relative to the 4-star boundary: the decision a plan faces."""
        if gap not in self.mapping.index or self.mapping.loc[gap, "cms_measure_id"] == "":
            return 0.0
        r = self.mapping.loc[gap]
        d = self.density.loc[r["cms_measure_id"]]
        pass_through = 1.0 if r["link"] == "direct" else float(d["mean_rate"])
        rho_value = float(d[f"rho_{rho}"]) if position is None else self._position_rho(r["cms_measure_id"], position)
        return (
            _end(QBP_PER_ENROLLEE_YEAR, qbp)
            * rho_value
            * pass_through
            / self._share(r["eligible_group"], share)
            / r["gaps_sharing_measure"]
        )

    def _position_rho(self, measure: str, position: str) -> float:
        if self.position_density is None:
            raise ValueError("position_density not loaded (config/cms_star_position_density.csv)")
        row = self.position_density[(self.position_density["measure_id"] == measure) & (self.position_density["position"] == position)]
        return float(row["rho_base"].iloc[0])

    def position_table(self, counts: dict[str, int] | None = None) -> pd.DataFrame:
        """USD per closure at base inputs for each gap that has a Star measure, by contract position, plus the gap-mix mean."""
        if self.position_density is None:
            return pd.DataFrame()
        rows = []
        for pos, g in self.position_density.groupby("position", sort=False):
            n = int(g["n_contracts"].max())
            vals = {gap: self.components(gap, position=pos) for gap in self.mapping.index if self.mapping.loc[gap, "cms_measure_id"] != ""}
            row = {"position": pos, "n_contracts": n} | {k: round(v, 1) for k, v in vals.items()}
            if counts:
                tot = sum(counts.values())
                row["mean_over_gap_mix"] = round(sum(c * self.components(gg, position=pos) for gg, c in counts.items()) / tot, 1)
            rows.append(row)
        return pd.DataFrame(rows)

    def share_sweep(self, counts: dict[str, int], group: str, shares: list[float]) -> pd.DataFrame:
        """Mean USD per closure over the gap mix as one group's eligible share moves (e.g. diabetes 0.20 to 0.50).
        break_even_scale = base mean / this mean: multiply the base break-even lift by it."""
        base = _mix(self, counts)
        out = []
        for sh in shares:
            m = StarValueModel(self.density.reset_index(), self.mapping.reset_index(), {**(self.plan_shares or {}), group: sh}, self.position_density)
            v = _mix(m, counts)
            out.append({"group": group, "eligible_share": sh, "mean_usd_per_closure": v, "break_even_scale": base / v if v else float("nan")})
        return pd.DataFrame(out)

    def value(self, gap: str, case: str = "base") -> float:
        return self.components(gap, case, case, case)

    def mean_value(self, gaps: pd.Series, case: str = "base") -> float:
        return float(np.mean([self.value(g, case) for g in gaps])) if len(gaps) else float("nan")

    def provenance_rows(self) -> list[dict]:
        """Empirical rho per measure, for the assumptions table in the report."""
        cls = Provenance.PUBLIC.value + " (computed from CMS 2026 Star Ratings data)"
        return [
            {
                "name": f"rho_{mid}_boundary_crossing_per_unit_rate",
                "low": float(r["rho_low"]),
                "base": float(r["rho_base"]),
                "high": float(r["rho_high"]),
                "unit": f"P(tip a 3.5-star contract to 4.0) per unit rate; CMS weight {int(r['weight'])}",
                "provenance": cls,
                "source": f"{int(r['n_contracts'])} MA contracts, CMS 2026 cut points and weights",
            }
            for mid, r in self.density.iterrows()
        ]


def _mix(model: StarValueModel, counts: dict[str, int]) -> float:
    n = sum(counts.values())
    return sum(c * model.value(g) for g, c in counts.items()) / n if n else float("nan")


def weight_discrepancies(incomm_weights_path: str, cms_weights_path: str, mapping: pd.DataFrame) -> pd.DataFrame:
    """InComm's StarsWeight for each gap's measure vs CMS's 2026 weight for the mapped Star measure."""
    inc = pd.read_csv(incomm_weights_path)
    cms = pd.read_csv(cms_weights_path).set_index("measure_id")["weight"]
    out = inc.merge(mapping[["care_gap_code", "cms_measure_id"]].rename(columns={"care_gap_code": "CareGapCode"}), on="CareGapCode")
    out["cms_2026_weight"] = out["cms_measure_id"].map(cms)
    out["note"] = np.where(out["cms_measure_id"].fillna("") == "", "not a 2026 Part C Star measure",
                           np.where(out["StarsWeight"] != out["cms_2026_weight"], "InComm weight differs from CMS", "matches"))
    return out[["CareGapCode", "HedisMeasureId", "StarsWeight", "cms_measure_id", "cms_2026_weight", "note"]]

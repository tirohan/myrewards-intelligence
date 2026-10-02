"""Build the empirical Stars -> QBP boundary density from CMS's PUBLIC 2026 Star Ratings data.

Why: the value of one extra care-gap closure through Star Ratings depends on how close real contracts sit to the
4-star QBP boundary. InComm cannot share data, so we measure it from the public CMS tables instead of assuming it.

Method (per measure m, for every Medicare Advantage contract with a rating):
  1. star(m, rate): from the published Part C cut points.
  2. A rate increase dr moves the measure up one star if rate + dr crosses the next cut point (exact, from data).
  3. That raises the contract's weighted-average rating by  w_m / W_c  (CMS weights; W_c = the contract's total
     rated weight). The published overall rating is rounded to half stars, so the raw rating is taken as uniform
     inside its band; a contract rated 3.5 crosses into the QBP at 4.0 with probability min(1, (w_m / W_c) / 0.5).
  4. rho_m = E_contracts[ P(cross) ] / dr, averaged over dr in {0.02, 0.03, 0.05} (range kept for sensitivity).
Enrollment cancels: expected QBP dollars = E x QBP$ x rho x (k / N) = QBP$ x rho x k / (N / E).

Validation: the same weights reproduce the published Part C summary rating from measure stars for the contracts
(see the JSON written beside the CSV); CMS's categorical adjustment and reward factor are not modelled, which is
the source of the small systematic gap.

Usage: python scripts/build_star_boundary_density.py [path/to/2026-star-ratings-data-tables.zip]
Downloads the public zip to data/cms/ if no path is given. Writes config/cms_star_boundary_density.csv,
config/cms_2026_measure_weights.csv and config/cms_star_engine_validation.json.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.cms.gov/files/zip/2026-star-ratings-data-tables.zip"
TECH_NOTES_URL = "https://www.cms.gov/files/document/2026-star-ratings-technical-notes.pdf"
DR_GRID = (0.02, 0.03, 0.05)
TARGET_MEASURES = ("C11", "C12", "C13", "C14")  # eye exam, blood sugar controlled, KED, controlling BP
# 2026 Technical Notes weights table (measure weights by category). C04/C05 are listed "1*".
WEIGHTS = {
    **{f"C{i:02d}": 1 for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 20, 21)},
    "C12": 3, "C14": 3, "C18": 3, **{f"C{i}": 2 for i in (22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33)}, "C30": 5,
    "D01": 2, "D02": 2, "D03": 2, "D04": 5, "D05": 2, "D06": 2, "D07": 1, "D08": 3, "D09": 3, "D10": 3,
    "D11": 1, "D12": 1,
}


def _measure_id(label: str) -> str | None:
    m = re.match(r"\s*([CD]\d\d)\*?:", label)
    return m.group(1) if m else None


def _read_csv(z: zipfile.ZipFile, key: str) -> list[list[str]]:
    name = next(n for n in z.namelist() if key in n and n.endswith(".csv"))
    return list(csv.reader(io.StringIO(z.read(name).decode("utf-8-sig", errors="replace"))))


def _pct(x: str) -> float:
    s = str(x).strip()
    try:
        return float(s.replace("%", "")) / 100 if "%" in s else float(s)
    except ValueError:
        return float("nan")


def main() -> None:
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "cms" / "2026-star-ratings-data-tables.zip"
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        zip_path.write_bytes(urllib.request.urlopen(req).read())  # noqa: S310 (fixed public CMS URL)
    z = zipfile.ZipFile(zip_path)

    def table(key: str) -> tuple[list[str | None], pd.DataFrame]:
        rows = _read_csv(z, key)
        return [_measure_id(n) for n in rows[2]], pd.DataFrame(rows[4:])

    ids_r, rates_raw = table("Measure Data")
    ids_s, stars_raw = table("Measure Stars")
    rates = pd.DataFrame({ids_r[i]: rates_raw[i].map(_pct) for i in range(len(ids_r)) if ids_r[i]})
    stars = pd.DataFrame(
        {ids_s[i]: pd.to_numeric(stars_raw[i].astype(str).str.strip(), errors="coerce") for i in range(len(ids_s)) if ids_s[i]}
    )
    contract = rates_raw[0].astype(str).str.strip()
    rates.insert(0, "contract", contract)
    stars.insert(0, "contract", contract)

    cp = _read_csv(z, "Part C Cut Points")
    cuts = {}
    for i, mid in enumerate(_measure_id(n) for n in cp[2]):
        if mid and mid.startswith("C"):
            pct = "%" in cp[5][i]
            vals = []
            for r in range(5, 9):  # lower bounds of 2, 3, 4, 5 stars
                mm = re.search(r">=\s*(-?[\d.]+)", cp[r][i])
                vals.append(float(mm.group(1)) / (100 if pct else 1) if mm else float("nan"))
            cuts[mid] = vals

    sname = next(n for n in z.namelist() if "Summary Ratings" in n and n.endswith(".xlsx"))
    summ = pd.read_excel(io.BytesIO(z.read(sname)), header=1).rename(columns={"Contract Number": "contract"})
    summ["contract"] = summ["contract"].astype(str).str.strip()
    d = stars.merge(summ[["contract", "2026 Part C Summary", "2026 Part D Summary", "2026 Overall"]], on="contract")
    d = d.merge(rates, on="contract", suffixes=("", "_rate"))
    part_d = pd.to_numeric(d["2026 Part D Summary"], errors="coerce").notna()
    d["rating"] = np.where(
        part_d, pd.to_numeric(d["2026 Overall"], errors="coerce"), pd.to_numeric(d["2026 Part C Summary"], errors="coerce")
    )
    d["is_pd"] = part_d
    sc = [c for c in stars.columns if c in WEIGHTS]
    d["W"] = d.apply(lambda r: sum(WEIGHTS[c] for c in sc if (c.startswith("C") or r["is_pd"]) and pd.notna(r[c])), axis=1)

    # --- validation: reconstruct the published Part C summary from measure stars
    cm = [c for c in sc if c.startswith("C")]
    wc = np.array([WEIGHTS[c] for c in cm], float)
    v = d[cm].to_numpy(float)
    ok = ~np.isnan(v)
    raw_c = np.where(ok, v, 0.0) @ wc / (ok * wc).sum(axis=1)
    pub_c = pd.to_numeric(d["2026 Part C Summary"], errors="coerce").to_numpy()
    mask = ~np.isnan(pub_c) & ~np.isnan(raw_c)
    rounded = np.round(raw_c * 2) / 2
    ma = d[d["contract"].str.startswith(("H", "R")) & d["rating"].notna() & (d["W"] > 0)]
    validation = {
        "source": URL,
        "contracts_validated": int(mask.sum()),
        "part_c_summary_exact_match_after_half_star_rounding": float((rounded[mask] == pub_c[mask]).mean()),
        "part_c_summary_within_half_star": float((np.abs(pub_c[mask] - raw_c[mask]) <= 0.5).mean()),
        "mean_published_minus_reconstructed": float((pub_c[mask] - raw_c[mask]).mean()),
        "not_modelled": "categorical adjustment index and reward factor (explains the small positive gap)",
        "ma_contracts_rated": int(len(ma)),
        "share_rated_3_5": float((ma["rating"] == 3.5).mean()),
        "share_rated_4_or_more": float((ma["rating"] >= 4).mean()),
        "median_total_rated_weight": float(ma["W"].median()),
        "dr_grid": list(DR_GRID),
    }

    def star(mid: str, r: float) -> int:
        return 1 + sum(1 for t in cuts[mid] if r >= t)

    rows = []
    for mid in TARGET_MEASURES:
        sub = ma[ma[mid + "_rate"].notna()]
        rho = []
        for dr in DR_GRID:
            p = []
            for r in sub.itertuples(index=False):
                rate = getattr(r, mid + "_rate")
                step = star(mid, rate + dr) - star(mid, rate)
                p.append(min(1.0, WEIGHTS[mid] * step / r.W / 0.5) if (r.rating == 3.5 and step > 0) else 0.0)
            rho.append(float(np.mean(p)) / dr)
        rows.append(
            {
                "measure_id": mid,
                "weight": WEIGHTS[mid],
                "n_contracts": int(len(sub)),
                "mean_rate": float(sub[mid + "_rate"].mean()),
                "rho_low": min(rho),
                "rho_base": float(np.mean(rho)),
                "rho_high": max(rho),
            }
        )
    out = ROOT / "config"
    pd.DataFrame(rows).to_csv(out / "cms_star_boundary_density.csv", index=False)
    pd.DataFrame(sorted(WEIGHTS.items()), columns=["measure_id", "weight"]).assign(source=TECH_NOTES_URL).to_csv(
        out / "cms_2026_measure_weights.csv", index=False
    )
    (out / "cms_star_engine_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

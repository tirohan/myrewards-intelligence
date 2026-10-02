"""Reward-amount pools: the empirical distribution of InComm reward amounts per care gap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import resolve_path
from .domain_resolutions import CATALOG_DEFAULT_AMOUNTS


def load_amount_pools(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """care_gap_code -> (amounts, probabilities). Empty dict if the pool file is absent."""
    file = resolve_path(path)
    if not file.exists():
        return {}
    df = pd.read_csv(file)
    pools = {}
    for gap, g in df.groupby("care_gap_code"):
        w = g["weight"].to_numpy(dtype=float)
        pools[str(gap)] = (g["amount"].to_numpy(dtype=float), w / w.sum())
    return pools


def draw_amounts(
    gaps: pd.Series, pools: dict[str, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
) -> np.ndarray:
    """One amount per row. Gaps with no pool get the InComm catalog default (constant)."""
    out = gaps.map(CATALOG_DEFAULT_AMOUNTS).to_numpy(dtype=float)
    for gap, (amounts, probs) in pools.items():
        idx = np.flatnonzero(gaps.to_numpy() == gap)
        if len(idx):
            out[idx] = rng.choice(amounts, size=len(idx), p=probs)
    return out

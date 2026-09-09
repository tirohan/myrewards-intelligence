"""Diagnostic plots for overlap, volume, and targeting quadrants."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_propensity_plot(t: np.ndarray, propensity: np.ndarray, path: Path) -> Path:
    """Histogram of e(x) for treated vs control."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t).astype(int)
    e = np.asarray(propensity, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(e[t == 0], bins=20, alpha=0.6, label="Control", color="#4C78A8")
    ax.hist(e[t == 1], bins=20, alpha=0.6, label="Treated", color="#F58518")
    ax.set_xlabel("Propensity to treat e(x)")
    ax.set_ylabel("Rows")
    ax.set_title("Overlap diagnostic (Track S — research-simulated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_gap_contrast_plot(by_gap: list[dict], path: Path) -> Path:
    """Treated vs control closure by care gap — the identification story."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not by_gap:
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.text(0.5, 0.5, "No care-gap rows", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    labels = [str(row["care_gap_code"]) for row in by_gap]
    treated = [
        row.get("treated_closure") if row.get("treated_closure") is not None else 0 for row in by_gap
    ]
    control = [
        row.get("control_closure") if row.get("control_closure") is not None else 0 for row in by_gap
    ]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - width / 2, treated, width, label="Treated closure", color="#F58518")
    ax.bar(x + width / 2, control, width, label="Control closure", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Closure rate")
    ax.set_title("Descriptive contrast by care gap (not CATE)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_volume_curve(curve: pd.DataFrame, path: Path, title: str) -> Path:
    """M5-style volume-at-threshold curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(curve["threshold"], curve["flagged_share"], marker="o", color="#4C78A8")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Share flagged")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_quadrant_chart(counts: pd.DataFrame, path: Path) -> Path:
    """Four-quadrant targeting counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(counts["quadrant"], counts["n"], color="#54A24B")
    ax.set_ylabel("Rows")
    ax.set_title("Targeting quadrants (Track S)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

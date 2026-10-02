"""Per-run manifest so reruns never silently overwrite a comparison baseline."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .config import PROJECT_ROOT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"  # not a git checkout (e.g. shared volume copy)


def write_run_manifest(
    name: str,
    config: dict,
    inputs: list[Path],
    outputs: list[Path],
    metrics: dict,
) -> Path:
    """Write runs/<ts>_<name>/manifest.json and copy small outputs beside it."""
    run_dir = PROJECT_ROOT / "runs" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for out in outputs:
        if out.exists():
            shutil.copy2(out, run_dir / out.name)
    manifest = {
        "name": name,
        "git_sha": _git_sha(),
        "config": config,
        "inputs": {str(p): _sha256(p) for p in inputs if p.exists()},
        "metrics": metrics,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path

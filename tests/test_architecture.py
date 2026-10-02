"""Hard rules from the project handoff, enforced as tests instead of prose."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from myrewards_intelligence.milestone6_reward_uplift.covariates import LEAKAGE_FEATURES
from myrewards_intelligence.milestone6_reward_uplift.evidence import TreatmentSource, m7_use

SRC = Path(__file__).resolve().parents[1] / "src" / "myrewards_intelligence"


def _imports(pkg: str) -> set[str]:
    found: set[str] = set()
    for path in (SRC / pkg).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
    return found


@pytest.mark.parametrize("pkg", ["milestone6_reward_uplift", "milestone7_claims_roi"])
def test_no_milestone5_import(pkg):
    assert not [m for m in _imports(pkg) if "milestone5" in m]


def test_m7_never_imports_m5_or_other_internals_besides_m6_evidence():
    allowed = {"milestone6_reward_uplift.domain_resolutions", "milestone6_reward_uplift.evidence"}
    m6 = {m.split("..")[-1] for m in _imports("milestone7_claims_roi") if "milestone6" in m}
    assert {m.removeprefix("myrewards_intelligence.") for m in m6} <= allowed | {""}


def test_closing_event_is_leakage():
    assert {"qualifying_claim_count", "days_since_last_qualifying_claim"} <= set(LEAKAGE_FEATURES)


def test_track_s_never_consumable_and_mock_research_only():
    assert m7_use(TreatmentSource.RESEARCH_SIMULATED) == "never"
    assert m7_use(TreatmentSource.RESEARCH_GENERATED_LEDGER) == "research-only"

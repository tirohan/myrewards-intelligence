"""Milestone 6 orchestrator entry point.

Track S reconstructs HEALTH_ACTION_REWARD assignment. Track R runs the same
learners on an IncentiveRewards-shaped ledger (research-generated mock when
the 3rd-party feed is absent). Never imports milestone5 packages.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from ..core.config import load_config, resolve_path, write_json
from ..core.logging import setup_logging
from ..milestone6_reward_uplift.ablation import export_ablation_scores, train_ablation_scorer
from ..milestone6_reward_uplift.covariates import prepare_covariates
from ..milestone6_reward_uplift.data import (
    LedgerNotAvailableError,
    headline_label_rows,
    load_analytical_dataset,
    load_track_r_bundle,
    load_treatment_events,
)
from ..milestone6_reward_uplift.derived_layer import generate_ddl_proposal, generate_scored_output
from ..milestone6_reward_uplift.elasticity import assess_elasticity
from ..milestone6_reward_uplift.evaluate_uplift import evaluate_uplift_multi_seed
from ..milestone6_reward_uplift.evidence import TreatmentSource
from ..milestone6_reward_uplift.overlap import assess_overlap
from ..milestone6_reward_uplift.plots import (
    save_gap_contrast_plot,
    save_propensity_plot,
    save_quadrant_chart,
    save_volume_curve,
)
from ..milestone6_reward_uplift.reports import write_all_reports
from ..milestone6_reward_uplift.targeting import (
    build_targeting_table,
    quadrant_counts,
    volume_curve,
)
from ..milestone6_reward_uplift.treatment import attach_treatment
from ..milestone6_reward_uplift.uplift import fit_uplift_models, scores_when_not_identified


def main() -> int:
    """Run the Milestone 6 pipeline using config.milestone6."""
    settings = load_config()
    cfg = settings.milestone6
    logger = setup_logging(
        log_level=settings.logging.level,
        log_dir=settings.paths.logs_dir,
        log_name="milestone6",
    )
    logger.info("MILESTONE 6: Reward Uplift / Elasticity (Track %s)", cfg.track)

    try:
        raw = load_analytical_dataset(settings)
        df = headline_label_rows(raw)
        track = (cfg.track or "S").upper()
        eligibility = None
        assumptions: dict[str, float | int | str] = {}
        if track == "R":
            events, eligibility, assumptions = load_track_r_bundle(settings, df)
            source = TreatmentSource.RESEARCH_GENERATED_LEDGER
            window_days = cfg.observation_window_days
        else:
            events = load_treatment_events(settings, df)
            source = TreatmentSource.RESEARCH_SIMULATED
            window_days = None
        logger.info(
            "Treatment events: n=%d source=%s grain=%s",
            len(events),
            events["source"].iloc[0] if len(events) else "none",
            cfg.treatment_grain,
        )
        treated = attach_treatment(
            df,
            treatment_source=source,
            events=events,
            eligibility=eligibility,
            window_days=window_days,
        )
        X, _ = prepare_covariates(treated, exclude_reward_features=True)
        overlap = assess_overlap(
            X,
            treated["t_issued"].to_numpy(),
            treated["y"].to_numpy(),
            min_arm_n=cfg.min_arm_n,
            min_ess_ratio=cfg.min_ess_ratio,
        )
        results = evaluate_uplift_multi_seed(
            treated,
            seeds=list(cfg.diagnostic_seeds),
            test_size=cfg.test_size,
            split_mode=cfg.split_mode,
        )
        results["track"] = track
        results["m7_may_consume"] = (
            treated["treatment_source_basis"].iloc[0] == TreatmentSource.INCOMM_ISSUED.value
            if len(treated)
            else False
        )
        results["observation_window_applied"] = bool(
            eligibility is not None and len(eligibility) > 0 and window_days is not None
        )
        results["mock_assumptions"] = assumptions
        elasticity = assess_elasticity(treated)
        ablation = train_ablation_scorer(X, treated["y"].to_numpy())
        export_ablation_scores(treated, ablation, cfg.ablation_scores_path)

        if results["cate_identified"]:
            fitted = fit_uplift_models(treated, X, random_state=cfg.random_state)
        else:
            fitted = scores_when_not_identified(
                n=len(treated),
                feature_names=list(X.columns),
                propensity=overlap.propensity,
                treated_rate=overlap.treated_closure_rate,
                ablation_mu0=ablation.scores,
                row_attributions=ablation.row_attributions,
            )
        ranked = build_targeting_table(
            treated,
            fitted,
            cate_identified=bool(results["cate_identified"]),
            ablation_scores=ablation.scores,
            tau_treat=cfg.uplift_tiers.tau_treat,
            propensity_high=cfg.uplift_tiers.propensity_high,
        )
        vol_tau = volume_curve(ranked["uplift_score"].to_numpy(), list(cfg.volume_cutpoints))
        vol_ablation = volume_curve(
            ranked["control_outcome_score"].to_numpy(), list(cfg.volume_cutpoints)
        )
        qcounts = quadrant_counts(ranked)

        fig_dir = resolve_path(settings.paths.reports_dir) / "figures"
        track_label = f"Track {track}"
        figures = {
            "propensity": save_propensity_plot(
                treated["t_issued"].to_numpy(), overlap.propensity, fig_dir / "propensity.png"
            ),
            "gap_contrast": save_gap_contrast_plot(
                results.get("by_care_gap") or [], fig_dir / "gap_contrast.png"
            ),
            "volume_tau": save_volume_curve(
                vol_tau, fig_dir / "volume_tau.png", f"Volume at τ threshold ({track_label})"
            ),
            "volume_ablation": save_volume_curve(
                vol_ablation,
                fig_dir / "volume_ablation.png",
                "Volume at ablation P(close) (not uplift)",
            ),
            "quadrant": save_quadrant_chart(qcounts, fig_dir / "quadrants.png"),
        }

        ddl = generate_ddl_proposal()
        scored = generate_scored_output(
            treated,
            fitted,
            cate_identified=bool(results["cate_identified"]),
            observation_window_days=cfg.observation_window_days,
        )

        models_dir = resolve_path(settings.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "der_MemberRewardUpliftScores.sql").write_text(ddl, encoding="utf-8")
        suffix = "track_r" if track == "R" else "track_s"
        scored.to_csv(models_dir / f"scored_uplift_{suffix}.csv", index=False)
        ranked.to_csv(models_dir / f"targeting_{suffix}.csv", index=False)
        vol_tau.to_csv(models_dir / "volume_tau.csv", index=False)
        write_json(
            {
                "completed": datetime.now(UTC).isoformat(),
                "track": cfg.track,
                "overlap_reasons": overlap.reasons,
                "cate_identified": results["cate_identified"],
                "split_mode": cfg.split_mode,
                "treatment_grain": cfg.treatment_grain,
                "treatment_source_basis": results.get("treatment_source_basis"),
                "m7_may_consume": results["m7_may_consume"],
                "observation_window_days": cfg.observation_window_days,
                "mock_assumptions": assumptions,
            },
            models_dir / "milestone6_metadata.json",
        )
        events_name = "incentive_rewards_events.csv" if track == "R" else "reconstructed_assignment_events.csv"
        events.to_csv(models_dir / events_name, index=False)

        written = write_all_reports(
            results=results,
            elasticity=elasticity,
            ranked=ranked,
            ddl=ddl,
            settings=settings,
            figures=figures,
            volume_tau=vol_tau,
            volume_ablation=vol_ablation,
            ablation_importance=ablation.global_importance,
        )
        logger.info("Reports: %s", written)
        logger.info(
            "MILESTONE 6 COMPLETE (Track %s). CATE identified=%s claim=%s m7_may_consume=%s",
            cfg.track,
            results["cate_identified"],
            results["headline_claim"],
            results["m7_may_consume"],
        )
        return 0
    except LedgerNotAvailableError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Milestone 6 failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

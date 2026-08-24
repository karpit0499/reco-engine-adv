from __future__ import annotations

import importlib.metadata
import platform
import time
from datetime import datetime, timezone

import numpy as np

from .bootstrap import confidence_intervals, paired_differences
from .dataset import EvaluationData
from .evaluation import (
    PerUserEvaluation,
    evaluate_recommendations,
    recommend_full_catalog,
    subgroup_metrics,
)
from .hashing import benchmark_source_hash, canonical_hash
from .models import (
    ALSModel,
    BPRModel,
    ItemKNNModel,
    PopularityModel,
    RecommenderModel,
    RetrievalRerankerModel,
    TwoTowerModel,
)
from .paths import BENCHMARK_ROOT


def environment_record() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _construct_model(
    spec: dict,
    fitted: dict[str, RecommenderModel],
) -> RecommenderModel:
    name = spec["name"]
    params = dict(spec.get("params", {}))
    if name == "popularity":
        return PopularityModel(**params)
    if name == "item_knn":
        return ItemKNNModel(**params)
    if name == "als":
        return ALSModel(**params)
    if name == "bpr":
        return BPRModel(**params)
    if name == "two_tower":
        return TwoTowerModel(**params)
    if name == "retrieval_reranker":
        if "als" not in fitted or "item_knn" not in fitted:
            raise ValueError(
                "retrieval_reranker must follow als and item_knn in the config"
            )
        return RetrievalRerankerModel(
            retrieval=fitted["als"], item_knn=fitted["item_knn"], **params
        )
    raise ValueError(f"unknown model: {name}")


def required_model_names(evaluated_models: set[str]) -> set[str]:
    required = set(evaluated_models)
    if "retrieval_reranker" in required:
        required.update({"item_knn", "als"})
    return required


def _coverage_points(evaluation: PerUserEvaluation) -> dict[str, float]:
    return {
        name: value
        for name, value in evaluation.point_metrics.items()
        if name.startswith("catalog_coverage_at_")
    }


def run_models(
    data: EvaluationData,
    protocol: dict,
    config: dict,
    evaluated_models: set[str] | None = None,
) -> tuple[dict, dict[str, PerUserEvaluation]]:
    cutoffs = [int(value) for value in protocol["evaluation"]["cutoffs"]]
    top_k = max(cutoffs)
    bootstrap = protocol["evaluation"]["bootstrap"]
    seed = int(config["seed"])
    batch_users = int(config["batch_users"])
    declared_names = [spec["name"] for spec in config["models"]]
    if len(declared_names) != len(set(declared_names)):
        raise ValueError("model names must be unique")
    evaluate = set(declared_names) if evaluated_models is None else set(evaluated_models)
    unknown = evaluate.difference(declared_names)
    if unknown:
        raise ValueError(f"requested models are absent from config: {sorted(unknown)}")
    required = required_model_names(evaluate)

    fitted: dict[str, RecommenderModel] = {}
    evaluations: dict[str, PerUserEvaluation] = {}
    model_results: dict[str, dict] = {}

    for spec in config["models"]:
        name = spec["name"]
        if name not in required:
            continue
        print(f"[{data.stage}] fitting {name}", flush=True)
        model = _construct_model(spec, fitted)
        if isinstance(model, RetrievalRerankerModel):
            model.candidate_seen_matrix = data.seen_matrix
        started = time.perf_counter()
        model.fit(data.fit_matrix, seed)
        fit_seconds = time.perf_counter() - started
        fitted[name] = model

        if name not in evaluate:
            continue
        started = time.perf_counter()
        recommendations = recommend_full_catalog(
            model, data, top_k=top_k, batch_users=batch_users
        )
        score_seconds = time.perf_counter() - started
        evaluation = evaluate_recommendations(recommendations, data, cutoffs)
        evaluations[name] = evaluation
        intervals = confidence_intervals(
            evaluation,
            cutoffs=cutoffs,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
        )
        diagnostics = {}
        if isinstance(model, RetrievalRerankerModel):
            diagnostics = {
                "retrieval_k": model.retrieval_k,
                "retrieval_target_recall": model.retrieval_target_recall(
                    data.evaluation_user_indices, data.targets
                ),
            }
        model_results[name] = {
            "params": dict(spec.get("params", {})),
            "diagnostic_fit_seconds": fit_seconds,
            "diagnostic_full_catalog_scoring_seconds": score_seconds,
            "metrics": intervals,
            "descriptive_metrics": _coverage_points(evaluation),
            "target_segments": {
                "warm_item": subgroup_metrics(evaluation, evaluation.warm_targets),
                "cold_item": subgroup_metrics(evaluation, ~evaluation.warm_targets),
            },
            "user_cohorts": {
                "warm": subgroup_metrics(evaluation, ~evaluation.cold_users),
                "cold": subgroup_metrics(evaluation, evaluation.cold_users),
            },
            "retrieval_diagnostics": diagnostics or None,
        }
        primary = protocol["evaluation"]["primary_metric"]
        print(
            f"[{data.stage}] {name}: {primary}="
            f"{evaluation.point_metrics[primary]:.6f}",
            flush=True,
        )

    if set(evaluations) != evaluate:
        raise ValueError("not every requested model was evaluated")
    result = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": data.stage,
        "test_accessed": data.stage == "test",
        "archive_sha256": data.archive_sha256,
        "protocol_sha256": data.protocol_hash,
        "config_sha256": canonical_hash(config),
        "benchmark_code_sha256": benchmark_source_hash(BENCHMARK_ROOT),
        "split_artifact_sha256": data.split_artifact_sha256,
        "split_manifest_sha256": data.split_manifest_sha256,
        "environment": environment_record(),
        "dataset": data.summary(),
        "uncertainty_notes": {
            "unit": "evaluation user",
            "catalog_coverage": (
                "Catalog coverage is a non-additive set-union statistic and is reported "
                "as a full-cohort descriptive estimate without a percentile confidence interval."
            ),
            "runtime": (
                "Wall-clock fit and full-catalog scoring times are local diagnostics only; "
                "they are not endpoint, browser, serving-throughput, or ANN latency claims."
            ),
        },
        "models": model_results,
    }
    return result, evaluations


def recompute_selection(result: dict, protocol: dict, config: dict) -> tuple[str, str]:
    primary = protocol["evaluation"]["primary_metric"]
    declared_order = [spec["name"] for spec in config["models"]]
    if set(result["models"]) != set(declared_order):
        raise ValueError("validation artifact does not contain every declared model")
    selected = max(
        declared_order,
        key=lambda name: (
            result["models"][name]["metrics"][primary]["estimate"],
            -declared_order.index(name),
        ),
    )
    simple_names = protocol["evaluation"]["simple_baselines"]
    strongest_simple = max(
        simple_names,
        key=lambda name: (
            result["models"][name]["metrics"][primary]["estimate"],
            -declared_order.index(name),
        ),
    )
    return selected, strongest_simple


def add_validation_selection(
    result: dict,
    evaluations: dict[str, PerUserEvaluation],
    protocol: dict,
    config: dict,
) -> dict:
    primary = protocol["evaluation"]["primary_metric"]
    selected, strongest_simple = recompute_selection(result, protocol, config)
    bootstrap = protocol["evaluation"]["bootstrap"]
    cutoffs = [int(value) for value in protocol["evaluation"]["cutoffs"]]
    comparisons = {}
    for name, evaluation in evaluations.items():
        comparisons[name] = paired_differences(
            evaluation,
            evaluations[strongest_simple],
            cutoffs=cutoffs,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
        )
    result["selection"] = {
        "rule": protocol["selection"]["rule"],
        "primary_metric": primary,
        "selected_model": selected,
        "strongest_simple_baseline": strongest_simple,
        "paired_differences_vs_strongest_simple": comparisons,
    }
    return result


def add_test_comparison(
    result: dict,
    evaluations: dict[str, PerUserEvaluation],
    receipt: dict,
    protocol: dict,
) -> dict:
    selected = receipt["selected_model"]
    baseline = receipt["strongest_simple_baseline"]
    selected_evaluation = evaluations[selected]
    baseline_evaluation = evaluations[baseline]
    bootstrap = protocol["evaluation"]["bootstrap"]
    cutoffs = [int(value) for value in protocol["evaluation"]["cutoffs"]]
    arguments = {
        "cutoffs": cutoffs,
        "replicates": int(bootstrap["replicates"]),
        "seed": int(bootstrap["seed"]),
        "confidence_level": float(bootstrap["confidence_level"]),
    }
    overall = paired_differences(
        selected_evaluation, baseline_evaluation, **arguments
    )
    cohorts = {}
    for name, mask in (
        ("warm", ~selected_evaluation.cold_users),
        ("cold", selected_evaluation.cold_users),
    ):
        if np.any(mask):
            cohorts[name] = paired_differences(
                selected_evaluation, baseline_evaluation, mask=mask, **arguments
            )
        else:
            cohorts[name] = None
    descriptive = {
        name: selected_evaluation.point_metrics[name]
        - baseline_evaluation.point_metrics[name]
        for name in selected_evaluation.point_metrics
        if name.startswith("catalog_coverage_at_")
    }
    primary = receipt["primary_metric"]
    result["frozen_test_comparison"] = {
        "selected_model": selected,
        "strongest_simple_baseline": baseline,
        "paired_differences": overall,
        "cohort_paired_differences": cohorts,
        "descriptive_catalog_coverage_differences": descriptive,
        "acceptance_gate": {
            "rule": protocol["selection"]["test_acceptance_gate"],
            "metric": primary,
            "passed": overall[primary]["ci_low"] > 0.0,
        },
    }
    return result

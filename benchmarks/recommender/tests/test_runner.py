from __future__ import annotations

import numpy as np

from reco_benchmark.evaluation import PerUserEvaluation
from reco_benchmark.runner import add_test_comparison, required_model_names


def _evaluation(value: float) -> PerUserEvaluation:
    recommendations = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    values = {
        "recall_at_10": np.full(4, value),
        "ndcg_at_10": np.full(4, value),
        "mrr_at_10": np.full(4, value),
        "user_coverage_at_10": np.ones(4),
    }
    point = {name: float(array.mean()) for name, array in values.items()}
    point["catalog_coverage_at_10"] = 0.5 + value / 10
    return PerUserEvaluation(
        recommendations=recommendations,
        values=values,
        point_metrics=point,
        warm_targets=np.ones(4, dtype=bool),
        cold_users=np.asarray([False, False, True, True]),
        catalog_size=20,
    )


def test_frozen_test_compares_selected_and_baseline_with_strict_gate() -> None:
    selected = _evaluation(1.0)
    baseline = _evaluation(0.0)
    protocol = {
        "evaluation": {
            "cutoffs": [10],
            "bootstrap": {
                "replicates": 100,
                "seed": 7,
                "confidence_level": 0.95,
            },
        },
        "selection": {"test_acceptance_gate": "lower bound > 0"},
    }
    receipt = {
        "selected_model": "als",
        "strongest_simple_baseline": "item_knn",
        "primary_metric": "ndcg_at_10",
    }
    result = {}
    add_test_comparison(
        result,
        {"als": selected, "item_knn": baseline},
        receipt,
        protocol,
    )
    comparison = result["frozen_test_comparison"]
    assert comparison["acceptance_gate"]["passed"] is True
    assert comparison["paired_differences"]["ndcg_at_10"] == {
        "estimate": 1.0,
        "ci_low": 1.0,
        "ci_high": 1.0,
    }
    assert comparison["cohort_paired_differences"]["cold"] is not None
    assert np.isclose(
        comparison["descriptive_catalog_coverage_differences"][
            "catalog_coverage_at_10"
        ],
        0.1,
    )


def test_two_stage_dependencies_are_frozen_with_requested_models() -> None:
    assert required_model_names({"retrieval_reranker", "popularity"}) == {
        "retrieval_reranker",
        "als",
        "item_knn",
        "popularity",
    }

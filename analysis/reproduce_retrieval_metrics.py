"""Reproduce latent-category checks from the sanitized saved-model fixture.

The check answers whether the saved synthetic embedding space recovers each user's
hidden preferred category. It is not a held-out recommendation evaluation: training
used every positive event and recorded no split, so conventional Recall@K/NDCG@K is
unavailable.

By default this script needs no GCP credentials. ``--refresh-from-gcp`` rebuilds the
sanitized fixture with read-only cloud queries. ``--index-endpoint`` optionally probes
Vertex Vector Search directly; it never calls the API route that logs impressions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from metrics_fixture import (
    CUTOFF,
    CUTOFF_IMPRESSION_ID,
    DEFAULT_FIXTURE,
    PROJECT,
    fixture_fingerprint,
    load_fixture,
    refresh_fixture_from_gcp,
)


DEFAULT_REGION = "europe-west3"


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo))


def latency_summary(values_ms: list[float]) -> dict[str, Any]:
    return {
        "successful_queries": len(values_ms),
        "min_ms": min(values_ms),
        "p50_ms": statistics.median(values_ms),
        "p95_ms": percentile(values_ms, 0.95),
        "max_ms": max(values_ms),
    }


def git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_name() -> str | None:
    if sys.platform != "darwin":
        return platform.processor() or None
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or None


def load_retrieval_arrays(fixture: dict[str, Any]) -> tuple[
    list[str], np.ndarray, dict[str, np.ndarray], dict[str, str], dict[str, str]
]:
    data = fixture["retrieval"]
    item_ids = list(data["item_ids"])
    item_matrix = np.asarray(data["item_embeddings"], dtype=np.float32)
    user_vectors = {
        user_id: np.asarray(vector, dtype=np.float32)
        for user_id, vector in zip(data["user_ids"], data["user_embeddings"])
    }
    truth = dict(zip(data["user_ids"], data["user_preferred_categories"]))
    catalog = dict(zip(item_ids, data["item_categories"]))
    dimension = int(data["dimension"])
    if item_matrix.shape != (len(item_ids), dimension):
        raise RuntimeError(f"item embedding matrix has unexpected shape {item_matrix.shape}")
    if any(vector.shape != (dimension,) for vector in user_vectors.values()):
        raise RuntimeError("one or more user vectors has the wrong dimension")
    return item_ids, item_matrix, user_vectors, truth, catalog


def score_rankings(
    rankings: dict[str, list[str]],
    truth: dict[str, str],
    catalog: dict[str, str],
    top_k: int,
) -> dict[str, Any]:
    discounts = [1.0 / math.log2(rank + 2) for rank in range(top_k)]
    precisions, recalls, ndcgs, expected_random = [], [], [], []
    hits = 0
    category_sizes = Counter(catalog.values())

    for user_id, ranked_ids in rankings.items():
        category = truth[user_id]
        relevance = [
            1 if catalog.get(item_id) == category else 0
            for item_id in ranked_ids[:top_k]
        ]
        if len(relevance) != top_k:
            raise RuntimeError(
                f"{user_id} returned {len(relevance)} results, expected {top_k}"
            )
        user_hits = sum(relevance)
        relevant_count = category_sizes[category]
        dcg = sum(value * discount for value, discount in zip(relevance, discounts))
        idcg = sum(discounts[: min(top_k, relevant_count)])
        hits += user_hits
        precisions.append(user_hits / top_k)
        recalls.append(user_hits / relevant_count)
        ndcgs.append(dcg / idcg if idcg else 0.0)
        expected_random.append(relevant_count / len(catalog))

    return {
        "users": len(rankings),
        "retrieved_slots": len(rankings) * top_k,
        "preferred_category_hits": hits,
        "preferred_category_precision_at_k_macro": statistics.fmean(precisions),
        "preferred_category_precision_at_k_micro": hits / (len(rankings) * top_k),
        "preferred_category_recall_at_k_macro": statistics.fmean(recalls),
        "preferred_category_ndcg_at_k_macro": statistics.fmean(ndcgs),
        "expected_random_precision_for_sample": statistics.fmean(expected_random),
    }


def exact_rankings(
    users: list[str],
    item_ids: list[str],
    item_matrix: np.ndarray,
    user_vectors: dict[str, np.ndarray],
    top_k: int,
) -> dict[str, list[str]]:
    rankings = {}
    for user_id in users:
        scores = item_matrix @ user_vectors[user_id]
        order = np.argsort(-scores, kind="stable")[:top_k]
        rankings[user_id] = [item_ids[int(index)] for index in order]
    return rankings


def benchmark_exact(
    users: list[str],
    item_matrix: np.ndarray,
    user_vectors: dict[str, np.ndarray],
    top_k: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    def query(user_id: str) -> None:
        scores = item_matrix @ user_vectors[user_id]
        np.argsort(-scores, kind="stable")[:top_k]

    for index in range(warmup):
        query(users[index % len(users)])
    latencies = []
    for index in range(repeats):
        started = time.perf_counter_ns()
        query(users[index % len(users)])
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)

    return {
        "status": "measured_non_serving_diagnostic",
        "operation": "NumPy exact dot-product over every item plus stable full sort",
        "warning": "in-process diagnostic; not ANN, network, or production serving latency",
        "catalog_items": int(item_matrix.shape[0]),
        "dimension": int(item_matrix.shape[1]),
        "top_k": top_k,
        "warmup_queries": warmup,
        "timed_queries": repeats,
        **latency_summary(latencies),
    }


def benchmark_vector_search(
    endpoint_name: str,
    deployed_index_id: str,
    region: str,
    project: str,
    users: list[str],
    user_vectors: dict[str, np.ndarray],
    truth: dict[str, str],
    catalog: dict[str, str],
    top_k: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
    deployed = [
        {"id": item.id, "index": item.index}
        for item in endpoint.gca_resource.deployed_indexes
    ]
    base = {
        "endpoint": endpoint_name,
        "deployed_index_id": deployed_index_id,
        "public_endpoint_domain": endpoint.gca_resource.public_endpoint_domain_name,
        "deployed_indexes": deployed,
    }
    if deployed_index_id not in {item["id"] for item in deployed}:
        return {
            **base,
            "status": "unavailable",
            "reason": f"{deployed_index_id!r} is not deployed on this endpoint",
        }

    try:
        for index in range(warmup):
            user_id = users[index % len(users)]
            endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=[user_vectors[user_id].tolist()],
                num_neighbors=top_k,
            )

        latencies, rankings = [], {}
        for index in range(repeats):
            user_id = users[index % len(users)]
            started = time.perf_counter_ns()
            response = endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=[user_vectors[user_id].tolist()],
                num_neighbors=top_k,
            )
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            if user_id not in rankings:
                rankings[user_id] = [neighbor.id for neighbor in response[0]]

        result = {
            **base,
            "status": "measured",
            "measurement_point": "client laptop to public Vertex AI Vector Search endpoint",
            "top_k": top_k,
            "warmup_queries": warmup,
            "timed_queries": repeats,
            **latency_summary(latencies),
        }
        if len(rankings) == len(users):
            result["category_metrics"] = score_rankings(rankings, truth, catalog, top_k)
        return result
    except Exception as error:
        return {
            **base,
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error),
            "warmup_queries_requested": warmup,
            "timed_queries_requested": repeats,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--cutoff",
        default=CUTOFF,
        help="experiment cutoff used only with --refresh-from-gcp",
    )
    parser.add_argument(
        "--cutoff-impression-id",
        default=CUTOFF_IMPRESSION_ID,
        help="stable tie-break at the exact cutoff, used only when refreshing",
    )
    parser.add_argument(
        "--refresh-from-gcp",
        action="store_true",
        help="replace the sanitized fixture using read-only queries, then evaluate it",
    )
    parser.add_argument("--project", default=os.environ.get("PROJECT", PROJECT))
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--sample-users",
        type=int,
        default=0,
        help="exact-check user count; 0 means every fixture user (default)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--local-warmup", type=int, default=100)
    parser.add_argument("--local-repeats", type=int, default=2_000)
    parser.add_argument("--index-endpoint")
    parser.add_argument("--deployed-index-id", default="items_v1")
    parser.add_argument("--online-sample-users", type=int, default=50)
    parser.add_argument("--online-warmup", type=int, default=5)
    parser.add_argument("--online-repeats", type=int, default=50)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_users < 0 or min(args.top_k, args.local_repeats) <= 0:
        raise SystemExit("sample-users must be non-negative; top-k/repeats must be positive")
    if args.refresh_from_gcp:
        refresh_fixture_from_gcp(
            args.fixture,
            project=args.project,
            cutoff=args.cutoff,
            cutoff_impression_id=args.cutoff_impression_id,
        )
    fixture = load_fixture(args.fixture)
    item_ids, item_matrix, user_vectors, truth, catalog = load_retrieval_arrays(fixture)

    available_users = sorted(set(user_vectors) & set(truth))
    users = (
        available_users
        if args.sample_users == 0
        else available_users[: args.sample_users]
    )
    if not users:
        raise SystemExit("no users have both an embedding and synthetic ground truth")
    if args.sample_users and len(users) != args.sample_users:
        raise SystemExit(f"only {len(users)} users are available")

    rankings = exact_rankings(users, item_ids, item_matrix, user_vectors, args.top_k)
    exact_metrics = score_rankings(rankings, truth, catalog, args.top_k)
    positive = fixture["source"]["positive_interactions"]
    trained_at = fixture["source"]["user_embeddings"]["trained_at"]
    all_positives_predate_training = positive["last_event_at"] < trained_at

    result: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "url": "https://github.com/karpit0499/reco-engine-adv",
            "commit_role": "repository/model state inspected before audit edits",
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
        },
        "environment": {
            "python": platform.python_version(),
            "packages": {"numpy": importlib.metadata.version("numpy")},
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": cpu_name(),
        },
        "invocation": ["python", *sys.argv],
        "fixture": fixture_fingerprint(args.fixture),
        "fixture_refreshed_from_gcp_this_run": args.refresh_from_gcp,
        "source_snapshot": fixture["source"],
        "protocol": {
            "users": (
                f"all {len(users)} fixture users"
                if len(users) == len(available_users)
                else f"lexicographically first {len(users)} fixture users"
            ),
            "candidate_set": f"all {len(item_ids)} fixture items",
            "ranking": "exact dot product, descending; fixture item order breaks ties",
            "top_k": args.top_k,
            "binary_relevance": "1 iff item.category == user.preferred_category; else 0",
            "split": "none; saved model was trained on every positive interaction",
            "synthetic_ground_truth": True,
            "interpretation": "latent category recovery / saved-space consistency only",
        },
        "exact_fixture_artifacts": {
            "category_metrics": exact_metrics,
            "local_latency": benchmark_exact(
                users,
                item_matrix,
                user_vectors,
                args.top_k,
                args.local_warmup,
                args.local_repeats,
            ),
        },
        "held_out_interaction_metrics": {
            "status": "unavailable",
            "recall_at_k": None,
            "ndcg_at_k": None,
            "all_positive_events_predate_latest_training": all_positives_predate_training,
            "reason": (
                "training/train.py queries every click/add_to_cart/purchase without a split, "
                "and the saved artifacts contain no recorded held-out positives. Reporting "
                "conventional held-out Recall@K or NDCG@K would be data leakage."
            ),
        },
    }

    if args.index_endpoint:
        online_users = users[: min(args.online_sample_users, len(users))]
        result["vector_search"] = benchmark_vector_search(
            args.index_endpoint,
            args.deployed_index_id,
            args.region,
            args.project,
            online_users,
            user_vectors,
            truth,
            catalog,
            args.top_k,
            args.online_warmup,
            args.online_repeats,
        )
    else:
        result["vector_search"] = {
            "status": "not_requested",
            "reason": "pass --index-endpoint to run the credentialed serving probe",
        }

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload)

    metric = exact_metrics
    print("Latent-category recovery (synthetic consistency; NOT held-out evaluation)")
    print(
        f"  {metric['preferred_category_hits']}/{metric['retrieved_slots']} relevant slots; "
        f"Precision@{args.top_k}={metric['preferred_category_precision_at_k_micro']:.4f}; "
        f"category-NDCG@{args.top_k}={metric['preferred_category_ndcg_at_k_macro']:.4f}"
    )
    print(
        f"  category-recall proxy@{args.top_k}="
        f"{metric['preferred_category_recall_at_k_macro']:.4f}; "
        f"random expected Precision={metric['expected_random_precision_for_sample']:.6f}"
    )
    print("Held-out interaction Recall@K/NDCG@K: UNAVAILABLE (no recorded split)")
    local = result["exact_fixture_artifacts"]["local_latency"]
    print(
        f"Local non-serving diagnostic ({local['catalog_items']} x {local['dimension']}, "
        f"n={local['timed_queries']}): p50={local['p50_ms']:.4f} ms, "
        f"p95={local['p95_ms']:.4f} ms"
    )
    online = result["vector_search"]
    if online["status"] == "measured":
        print(
            f"Vector Search (n={online['timed_queries']}): p50={online['p50_ms']:.1f} ms, "
            f"p95={online['p95_ms']:.1f} ms"
        )
    elif args.index_endpoint:
        print(f"Vector Search: UNAVAILABLE ({online.get('reason') or online.get('error')})")
    if args.json_out:
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

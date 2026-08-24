"""Evaluate one immutable, target-bound embedding generation without mutation."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, storage


ROOT = pathlib.Path(__file__).resolve().parents[1]
if (ROOT / "training").is_dir():
    sys.path.insert(0, str(ROOT / "training"))

from generation_contract import (  # noqa: E402
    ACTIVE_GENERATION_TABLE,
    USER_GENERATIONS_TABLE,
    ServingBinding,
    active_state_receipt_sha256,
    canonical_user_payload,
    item_blob_name,
    manifest_blob_name,
    manifest_uri,
    parse_item_payload,
    parse_user_payload,
    promotion_decision,
    sha256_bytes,
    user_blob_name,
    validate_binding,
    validate_generation_id,
    validate_manifest,
    validate_ops_image_uri,
    validate_sha256,
    validate_source_revision,
)


PROJECT = os.environ["PROJECT"]
BUCKET = os.environ.get("ARTIFACT_BUCKET", f"{PROJECT}-artifacts")
VECTOR_BUCKET = os.environ.get("VECTOR_BUCKET", f"{PROJECT}-vectors")
TOP_K = 10
N_USERS = 50
FLOOR = float(os.environ.get("RECALL_FLOOR", "0.35"))
MARGIN = float(os.environ.get("RECALL_MARGIN", "0.03"))


def conditional_download(blob, expected_generation: int | None = None) -> tuple[bytes, int]:
    """Reload metadata first, then bind the content read to that object generation."""
    blob.reload()
    generation = int(blob.generation)
    if expected_generation is not None and generation != expected_generation:
        raise RuntimeError("immutable GCS object generation no longer matches the manifest")
    return blob.download_as_bytes(if_generation_match=generation), generation


def active_champion(
    bq: bigquery.Client, gcs: storage.Client, binding: ServingBinding
) -> dict[str, object]:
    table_id = f"{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}"
    try:
        table = bq.get_table(table_id)
    except NotFound:
        return {
            "generation_id": "__uninitialized__",
            "epoch": 0,
            "champion_score": None,
            "manifest_object_generation": 0,
            "manifest_sha256": "__uninitialized__",
            "ops_image_uri": "__uninitialized__",
            "source_revision": "__uninitialized__",
        }
    schema = getattr(table, "schema", None)
    if schema is not None:
        fields = {field.name for field in schema}
        required_fields = {
            "manifest_object_generation",
            "manifest_sha256",
            "ops_image_uri",
            "source_revision",
        }
        if not required_fields.issubset(fields):
            raise RuntimeError(
                "legacy active-state schema requires the explicit receipt migration"
            )
    rows = list(
        bq.query(
            f"""SELECT generation_id, epoch, champion_score, manifest_uri,
                       manifest_object_generation, manifest_sha256,
                       ops_image_uri, source_revision,
                       index_name, index_endpoint, deployed_index_id
                FROM `{table_id}` WHERE singleton = TRUE"""
        ).result()
    )
    if not rows:
        return {
            "generation_id": "__uninitialized__",
            "epoch": 0,
            "champion_score": None,
            "manifest_object_generation": 0,
            "manifest_sha256": "__uninitialized__",
            "ops_image_uri": "__uninitialized__",
            "source_revision": "__uninitialized__",
        }
    if len(rows) != 1:
        raise RuntimeError(f"{table_id} must contain exactly one singleton row")
    row = rows[0]
    active_binding = validate_binding(
        row["index_name"], row["index_endpoint"], row["deployed_index_id"]
    )
    if active_binding != binding:
        raise RuntimeError("active state is bound to a different Vertex target")
    required = (
        "champion_score",
        "manifest_uri",
        "manifest_object_generation",
        "manifest_sha256",
        "ops_image_uri",
        "source_revision",
    )
    if any(row[field] is None for field in required):
        raise RuntimeError(
            "legacy active state has no immutable receipt; explicit migration is required"
        )
    generation_id = validate_generation_id(str(row["generation_id"]))
    epoch = int(row["epoch"])
    manifest_object_generation = int(row["manifest_object_generation"])
    manifest_sha256 = validate_sha256(str(row["manifest_sha256"]))
    ops_image_uri = validate_ops_image_uri(str(row["ops_image_uri"]))
    source_revision = validate_source_revision(str(row["source_revision"]))
    champion_score = float(row["champion_score"])
    if row["manifest_uri"] != manifest_uri(BUCKET, generation_id):
        raise RuntimeError("active manifest URI is malformed")
    active_manifest_payload, observed_generation = conditional_download(
        gcs.bucket(BUCKET).blob(manifest_blob_name(generation_id)),
        manifest_object_generation,
    )
    if (
        observed_generation != manifest_object_generation
        or sha256_bytes(active_manifest_payload) != manifest_sha256
    ):
        raise RuntimeError("active manifest differs from its immutable state receipt")
    validate_manifest(
        json.loads(active_manifest_payload),
        expected_generation_id=generation_id,
        canonical_bucket=BUCKET,
        regional_bucket=VECTOR_BUCKET,
        expected_binding=active_binding,
        expected_ops_image_uri=ops_image_uri,
        expected_source_revision=source_revision,
    )
    active_state_receipt_sha256(
        generation_id=generation_id,
        epoch=epoch,
        manifest_uri_value=str(row["manifest_uri"]),
        manifest_object_generation=manifest_object_generation,
        manifest_sha256=manifest_sha256,
        ops_image_uri=ops_image_uri,
        source_revision=source_revision,
        champion_score=champion_score,
        binding=active_binding,
    )
    return {
        "generation_id": generation_id,
        "epoch": epoch,
        "champion_score": champion_score,
        "manifest_object_generation": manifest_object_generation,
        "manifest_sha256": manifest_sha256,
        "ops_image_uri": ops_image_uri,
        "source_revision": source_revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--index-endpoint", required=True)
    parser.add_argument("--deployed-index-id", required=True)
    parser.add_argument("--ops-image-uri", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--decision-out", required=True)
    parser.add_argument("--score-out", required=True)
    parser.add_argument("--active-generation-out", required=True)
    parser.add_argument("--active-epoch-out", required=True)
    parser.add_argument("--manifest-generation-out", required=True)
    parser.add_argument("--manifest-sha256-out", required=True)
    parser.add_argument("--active-manifest-generation-out", required=True)
    parser.add_argument("--active-manifest-sha256-out", required=True)
    parser.add_argument("--active-ops-image-uri-out", required=True)
    parser.add_argument("--active-source-revision-out", required=True)
    parser.add_argument("--active-champion-score-out", required=True)
    args = parser.parse_args()
    generation_id = validate_generation_id(args.generation_id)
    binding = validate_binding(
        args.index_name, args.index_endpoint, args.deployed_index_id
    )
    ops_image_uri = validate_ops_image_uri(args.ops_image_uri)
    source_revision = validate_source_revision(args.source_revision)

    bq = bigquery.Client(project=PROJECT)
    gcs = storage.Client(project=PROJECT)
    canonical = gcs.bucket(BUCKET)
    regional = gcs.bucket(VECTOR_BUCKET)

    manifest_payload, manifest_object_generation = conditional_download(
        canonical.blob(manifest_blob_name(generation_id))
    )
    manifest_sha256 = sha256_bytes(manifest_payload)
    manifest = validate_manifest(
        json.loads(manifest_payload),
        expected_generation_id=generation_id,
        canonical_bucket=BUCKET,
        regional_bucket=VECTOR_BUCKET,
        expected_binding=binding,
        expected_ops_image_uri=ops_image_uri,
        expected_source_revision=source_revision,
    )
    dimensions = manifest["dimensions"]

    canonical_items, _ = conditional_download(
        canonical.blob(item_blob_name(generation_id)),
        manifest["items"]["canonical_object_generation"],
    )
    regional_items, _ = conditional_download(
        regional.blob(item_blob_name(generation_id)),
        manifest["items"]["regional_object_generation"],
    )
    expected_item_digest = manifest["items"]["sha256"]
    if (
        sha256_bytes(canonical_items) != expected_item_digest
        or sha256_bytes(regional_items) != expected_item_digest
    ):
        raise RuntimeError("candidate item copies no longer match their manifest")
    item_rows = parse_item_payload(canonical_items, dimensions=dimensions)
    if (
        len(item_rows) != manifest["items"]["count"]
        or manifest["items"]["canary_item_id"]
        not in {row["id"] for row in item_rows}
    ):
        raise RuntimeError("candidate item count no longer matches its manifest")

    users_payload, _ = conditional_download(
        canonical.blob(user_blob_name(generation_id)),
        manifest["users"]["object_generation"],
    )
    if sha256_bytes(users_payload) != manifest["users"]["sha256"]:
        raise RuntimeError("candidate user object no longer matches its manifest")
    user_rows = parse_user_payload(users_payload, dimensions=dimensions)
    if (
        len(user_rows) != manifest["users"]["count"]
        or manifest["users"]["canary_user_id"]
        not in {row["user_id"] for row in user_rows}
    ):
        raise RuntimeError("candidate user count no longer matches its manifest")

    query_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("generation_id", "STRING", generation_id)
        ]
    )
    staged_users = list(
        bq.query(
            f"""SELECT user_id, embedding
                FROM `{PROJECT}.reco.{USER_GENERATIONS_TABLE}`
                WHERE generation_id = @generation_id""",
            job_config=query_config,
        ).result()
    )
    staged_payload, _ = canonical_user_payload(staged_users, dimensions=dimensions)
    if sha256_bytes(staged_payload) != manifest["users"]["sha256"]:
        raise RuntimeError("BigQuery user staging differs from the immutable user object")

    item_ids = [row["id"] for row in item_rows]
    item_matrix = np.asarray([row["embedding"] for row in item_rows], dtype=np.float32)
    user_vectors = {
        row["user_id"]: np.asarray(row["embedding"], dtype=np.float32)
        for row in user_rows
    }
    truth = {
        row["user_id"]: row["preferred_category"]
        for row in bq.query(
            f"SELECT user_id, preferred_category FROM `{PROJECT}.reco.users_truth`"
        ).result()
    }
    catalog = {
        row["item_id"]: row["category"]
        for row in bq.query(
            f"SELECT item_id, category FROM `{PROJECT}.reco.catalog`"
        ).result()
    }
    sample = sorted(user_id for user_id in user_vectors if user_id in truth)[:N_USERS]
    if not sample:
        sys.exit("candidate has no users with synthetic ground truth")
    hits = total = 0
    for user_id in sample:
        top = np.argsort(-(item_matrix @ user_vectors[user_id]))[:TOP_K]
        hits += sum(catalog.get(item_ids[index]) == truth[user_id] for index in top)
        total += len(top)
    score = hits / total

    active = active_champion(bq, gcs, binding)
    active_generation = str(active["generation_id"])
    active_epoch = int(active["epoch"])
    champion = active["champion_score"]
    decision, reason = promotion_decision(
        score=score, floor=FLOOR, margin=MARGIN, champion=champion
    )
    print(
        f"generation {generation_id}: preferred-category share in top-{TOP_K} "
        f"= {score:.1%}; active={active_generation}@{active_epoch}; "
        f"champion={champion if champion is not None else 'none'}"
    )
    print(f"DECISION: {decision.upper()} — {reason}")

    outputs = (
        (args.decision_out, decision),
        (args.score_out, str(score)),
        (args.active_generation_out, active_generation),
        (args.active_epoch_out, str(active_epoch)),
        (args.manifest_generation_out, str(manifest_object_generation)),
        (args.manifest_sha256_out, manifest_sha256),
        (
            args.active_manifest_generation_out,
            str(active["manifest_object_generation"]),
        ),
        (args.active_manifest_sha256_out, str(active["manifest_sha256"])),
        (args.active_ops_image_uri_out, str(active["ops_image_uri"])),
        (args.active_source_revision_out, str(active["source_revision"])),
        (
            args.active_champion_score_out,
            str(champion if champion is not None else -1.0),
        ),
    )
    for path, value in outputs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value)


if __name__ == "__main__":
    main()

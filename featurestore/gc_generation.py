"""Explicit, fenced garbage collection for retired Vertex generations.

The command is dry-run unless ``--execute`` is supplied. It only removes
generation-namespaced Vertex datapoints; immutable GCS and BigQuery audit data
are retained. The active and immediately previous generations are never valid
targets.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from google.cloud import bigquery, storage


ROOT = pathlib.Path(__file__).resolve().parents[1]
for sibling in (ROOT / "training", ROOT / "pipelines"):
    if sibling.is_dir():
        sys.path.insert(0, str(sibling))

from generation_contract import (  # noqa: E402
    ACTIVE_GENERATION_TABLE,
    GENERATION_RETIREMENTS_TABLE,
    ManifestReceipt,
    ServingBinding,
    item_blob_name,
    manifest_blob_name,
    manifest_uri,
    namespaced_datapoint_id,
    parse_item_payload,
    sha256_bytes,
    validate_binding,
    validate_generation_id,
    validate_manifest,
    validate_ops_image_uri,
    validate_sha256,
    validate_source_revision,
)
from promote import (  # noqa: E402
    LeaseFence,
    acquire_lease,
    ensure_tables,
    heartbeat_lease,
    read_active_state,
    release_lease,
    state_receipt_sha256,
)


PROJECT = os.environ["PROJECT"]
REGION = os.environ.get("REGION", "europe-west3")
BUCKET = os.environ.get("ARTIFACT_BUCKET", f"{PROJECT}-artifacts")
VECTOR_BUCKET = os.environ.get("VECTOR_BUCKET", f"{PROJECT}-vectors")


@dataclass(frozen=True)
class RetirementRecord:
    generation_id: str
    retired_at: datetime
    drain_until: datetime
    retired_epoch: int
    successor_generation_id: str
    receipt: ManifestReceipt
    binding: ServingBinding
    ops_image_uri: str
    source_revision: str


def _timestamp(value, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise RuntimeError(f"retirement {field} is invalid")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_retirement_record(
    bq: bigquery.Client, generation_id: str
) -> RetirementRecord:
    generation_id = validate_generation_id(generation_id)
    rows = list(
        bq.query(
            f"""SELECT * FROM `{PROJECT}.reco.{GENERATION_RETIREMENTS_TABLE}`
                WHERE generation_id = @generation_id""",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "generation_id", "STRING", generation_id
                    )
                ]
            ),
        ).result()
    )
    if len(rows) != 1:
        raise RuntimeError(
            "generation GC requires exactly one immutable retirement record"
        )
    row = rows[0]
    try:
        retired_at = _timestamp(row["retired_at"], "timestamp")
        drain_until = _timestamp(row["drain_until"], "drain deadline")
        retired_epoch = int(row["retired_epoch"])
        successor = validate_generation_id(str(row["successor_generation_id"]))
        receipt = ManifestReceipt(
            uri=str(row["manifest_uri"]),
            object_generation=int(row["manifest_object_generation"]),
            sha256=validate_sha256(
                str(row["manifest_sha256"]), "retired manifest digest"
            ),
        )
        binding = validate_binding(
            row["index_name"], row["index_endpoint"], row["deployed_index_id"]
        )
        ops_image_uri = validate_ops_image_uri(str(row["ops_image_uri"]))
        source_revision = validate_source_revision(str(row["source_revision"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "legacy or malformed retirement record cannot authorize generation GC"
        ) from error
    if (
        retired_epoch <= 0
        or successor == generation_id
        or receipt.uri != manifest_uri(BUCKET, generation_id)
        or receipt.object_generation <= 0
        or drain_until <= retired_at
    ):
        raise RuntimeError("retirement record is internally inconsistent")
    return RetirementRecord(
        generation_id=generation_id,
        retired_at=retired_at,
        drain_until=drain_until,
        retired_epoch=retired_epoch,
        successor_generation_id=successor,
        receipt=receipt,
        binding=binding,
        ops_image_uri=ops_image_uri,
        source_revision=source_revision,
    )


def conditional_download(blob, expected_generation: int | None = None) -> bytes:
    blob.reload()
    generation = int(blob.generation)
    if expected_generation is not None and generation != expected_generation:
        raise RuntimeError("retired generation object changed after its manifest")
    return blob.download_as_bytes(if_generation_match=generation)


def validate_gc_target(
    *,
    state,
    retirement: RetirementRecord,
    generation_id: str,
    expected_active_generation: str,
    expected_active_epoch: int,
    binding: ServingBinding,
    now: datetime | None = None,
) -> None:
    """Pure fail-closed eligibility checks used by both dry-run and execution."""
    generation_id = validate_generation_id(generation_id)
    expected_active_generation = validate_generation_id(expected_active_generation)
    if state is None:
        raise RuntimeError("generation GC requires initialized active state")
    if (
        state.generation_id != expected_active_generation
        or state.epoch != expected_active_epoch
        or state.binding != binding
    ):
        raise RuntimeError("active generation/epoch/binding differs from the GC fence")
    if state.lease_owner or state.pending_generation_id or state.pending_manifest_uri:
        raise RuntimeError("generation GC refuses to overlap another remote operation")
    if generation_id == state.generation_id:
        raise RuntimeError("the active generation cannot be garbage-collected")
    if generation_id == state.previous_generation_id:
        raise RuntimeError("the retained rollback generation cannot be garbage-collected")
    if generation_id in state.retained_generation_ids:
        raise RuntimeError("a retained generation cannot be garbage-collected")
    if retirement.generation_id != generation_id:
        raise RuntimeError("retirement record names another generation")
    if retirement.binding != binding:
        raise RuntimeError("retirement record names another Vertex serving target")
    if retirement.retired_epoch > state.epoch:
        raise RuntimeError("retirement record is newer than the active state fence")
    current_time = _timestamp(
        now or datetime.now(timezone.utc), "validation timestamp"
    )
    if retirement.retired_at > current_time:
        raise RuntimeError("retirement timestamp is in the future")
    if retirement.drain_until > current_time:
        raise RuntimeError("retired generation is still inside its drain window")


def load_retired_generation(
    gcs: storage.Client,
    retirement: RetirementRecord,
    minimum_age: timedelta,
) -> list[str]:
    generation_id = retirement.generation_id
    bucket = gcs.bucket(BUCKET)
    manifest_payload = conditional_download(
        bucket.blob(manifest_blob_name(generation_id)),
        retirement.receipt.object_generation,
    )
    if sha256_bytes(manifest_payload) != retirement.receipt.sha256:
        raise RuntimeError("retired manifest failed its retirement-record digest")
    manifest = validate_manifest(
        json.loads(manifest_payload),
        expected_generation_id=generation_id,
        canonical_bucket=BUCKET,
        regional_bucket=VECTOR_BUCKET,
        expected_binding=retirement.binding,
        expected_ops_image_uri=retirement.ops_image_uri,
        expected_source_revision=retirement.source_revision,
    )
    created_at = datetime.fromisoformat(
        str(manifest["created_at"]).replace("Z", "+00:00")
    )
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - created_at < minimum_age:
        raise RuntimeError("generation is younger than the configured GC retention period")
    item_payload = conditional_download(
        bucket.blob(item_blob_name(generation_id)),
        manifest["items"]["canonical_object_generation"],
    )
    if sha256_bytes(item_payload) != manifest["items"]["sha256"]:
        raise RuntimeError("retired item artifact failed its manifest digest")
    rows = parse_item_payload(item_payload, dimensions=manifest["dimensions"])
    if len(rows) != manifest["items"]["count"]:
        raise RuntimeError("retired item count differs from its manifest")
    return [namespaced_datapoint_id(generation_id, row["id"]) for row in rows]


def fence_gc_target(
    bq: bigquery.Client, fence: LeaseFence, retirement: RetirementRecord
) -> None:
    """Recheck retention, drain, lease, and retirement receipts before mutation."""
    bq.query(
        f"""BEGIN TRANSACTION;
            ASSERT (SELECT COUNT(*) FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
              WHERE singleton = TRUE
                AND generation_id = @active_generation AND epoch = @active_epoch
                AND manifest_uri = @active_manifest_uri
                AND manifest_object_generation = @active_manifest_object_generation
                AND manifest_sha256 = @active_manifest_sha256
                AND ops_image_uri = @active_ops_image_uri
                AND source_revision = @active_source_revision
                AND champion_score = @active_champion_score
                AND index_name = @index_name AND index_endpoint = @index_endpoint
                AND deployed_index_id = @deployed_index_id
                AND lease_owner = @owner AND lease_epoch = @lease_epoch
                AND pending_generation_id = @target_generation
                AND pending_manifest_uri = @target_manifest_uri
                AND pending_manifest_object_generation = @target_manifest_generation
                AND pending_manifest_sha256 = @target_manifest_sha256
                AND pending_ops_image_uri = @target_ops_image_uri
                AND pending_source_revision = @target_source_revision
                AND pending_score = @lease_score
                AND lease_expires_at > CURRENT_TIMESTAMP()
                AND NOT @target_generation IN UNNEST(
                  IFNULL(retained_generation_ids, []))) = 1
              AS 'generation became retained or GC lease fence was lost';
            ASSERT (SELECT COUNT(*)
              FROM `{PROJECT}.reco.{GENERATION_RETIREMENTS_TABLE}`
              WHERE generation_id = @target_generation
                AND retired_at = @retired_at AND drain_until = @drain_until
                AND drain_until <= CURRENT_TIMESTAMP()
                AND retired_epoch = @retired_epoch
                AND successor_generation_id = @successor_generation
                AND manifest_uri = @target_manifest_uri
                AND manifest_object_generation = @target_manifest_generation
                AND manifest_sha256 = @target_manifest_sha256
                AND index_name = @index_name AND index_endpoint = @index_endpoint
                AND deployed_index_id = @deployed_index_id
                AND ops_image_uri = @target_ops_image_uri
                AND source_revision = @target_source_revision) = 1
              AS 'retirement receipt or drain fence changed';
            COMMIT TRANSACTION;""",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(name, field_type, value)
                for name, field_type, value in (
                    ("active_generation", "STRING", fence.active_generation),
                    ("active_epoch", "INT64", fence.active_epoch),
                    ("active_manifest_uri", "STRING", fence.active_manifest_uri),
                    (
                        "active_manifest_object_generation",
                        "INT64",
                        fence.active_manifest_object_generation,
                    ),
                    (
                        "active_manifest_sha256",
                        "STRING",
                        fence.active_manifest_sha256,
                    ),
                    ("active_ops_image_uri", "STRING", fence.active_ops_image_uri),
                    (
                        "active_source_revision",
                        "STRING",
                        fence.active_source_revision,
                    ),
                    (
                        "active_champion_score",
                        "FLOAT64",
                        fence.active_champion_score,
                    ),
                    ("index_name", "STRING", fence.binding.index_name),
                    ("index_endpoint", "STRING", fence.binding.index_endpoint),
                    (
                        "deployed_index_id",
                        "STRING",
                        fence.binding.deployed_index_id,
                    ),
                    ("owner", "STRING", fence.owner),
                    ("lease_epoch", "INT64", fence.lease_epoch),
                    ("target_generation", "STRING", retirement.generation_id),
                    ("target_manifest_uri", "STRING", retirement.receipt.uri),
                    (
                        "target_manifest_generation",
                        "INT64",
                        retirement.receipt.object_generation,
                    ),
                    (
                        "target_manifest_sha256",
                        "STRING",
                        retirement.receipt.sha256,
                    ),
                    (
                        "target_ops_image_uri",
                        "STRING",
                        retirement.ops_image_uri,
                    ),
                    (
                        "target_source_revision",
                        "STRING",
                        retirement.source_revision,
                    ),
                    ("lease_score", "FLOAT64", fence.candidate_score),
                    ("retired_at", "TIMESTAMP", retirement.retired_at),
                    ("drain_until", "TIMESTAMP", retirement.drain_until),
                    ("retired_epoch", "INT64", retirement.retired_epoch),
                    (
                        "successor_generation",
                        "STRING",
                        retirement.successor_generation_id,
                    ),
                )
            ]
        ),
    ).result()


def collect_generation(
    *,
    generation_id: str,
    expected_active_generation: str,
    expected_active_epoch: int,
    binding: ServingBinding,
    minimum_age_hours: float,
    execute: bool,
    batch_size: int = 100,
) -> None:
    if minimum_age_hours < 0 or batch_size <= 0:
        raise ValueError("GC age and batch size must be non-negative/positive")
    bq = bigquery.Client(project=PROJECT)
    gcs = storage.Client(project=PROJECT)
    ensure_tables(bq)
    state = read_active_state(bq)
    retirement = read_retirement_record(bq, generation_id)
    validate_gc_target(
        state=state,
        retirement=retirement,
        generation_id=generation_id,
        expected_active_generation=expected_active_generation,
        expected_active_epoch=expected_active_epoch,
        binding=binding,
    )
    datapoint_ids = load_retired_generation(
        gcs,
        retirement,
        timedelta(hours=minimum_age_hours),
    )
    action = "would remove" if not execute else "removing"
    print(f"{action} {len(datapoint_ids)} datapoints for {generation_id}")
    if not execute:
        print("dry run: pass --execute after reviewing the active fence")
        return

    owner = f"gc-{uuid.uuid4()}"
    fence = acquire_lease(
        bq,
        state=state,
        owner=owner,
        candidate_generation=generation_id,
        candidate_receipt=retirement.receipt,
        candidate_ops_image_uri=retirement.ops_image_uri,
        candidate_source_revision=retirement.source_revision,
        candidate_score=state.champion_score,
        evaluated_active_receipt_sha256=state_receipt_sha256(state),
    )
    try:
        heartbeat_lease(bq, fence, "gc-binding")
        fence_gc_target(bq, fence, retirement)
        from google.cloud import aiplatform

        aiplatform.init(project=PROJECT, location=REGION)
        index = aiplatform.MatchingEngineIndex(index_name=binding.index_name)
        for offset in range(0, len(datapoint_ids), batch_size):
            heartbeat_lease(bq, fence, "gc-removing")
            fence_gc_target(bq, fence, retirement)
            index.remove_datapoints(
                datapoint_ids=datapoint_ids[offset : offset + batch_size]
            )
        heartbeat_lease(bq, fence, "gc-complete")
        fence_gc_target(bq, fence, retirement)
    finally:
        release_lease(bq, fence)
    print(
        f"removed Vertex datapoints for {generation_id}; immutable GCS/BQ data retained"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--expected-active-generation", required=True)
    parser.add_argument("--expected-active-epoch", type=int, required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--index-endpoint", required=True)
    parser.add_argument("--deployed-index-id", required=True)
    parser.add_argument("--minimum-age-hours", type=float, default=168.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    collect_generation(
        generation_id=validate_generation_id(args.generation_id),
        expected_active_generation=args.expected_active_generation,
        expected_active_epoch=args.expected_active_epoch,
        binding=validate_binding(
            args.index_name, args.index_endpoint, args.deployed_index_id
        ),
        minimum_age_hours=args.minimum_age_hours,
        execute=args.execute,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

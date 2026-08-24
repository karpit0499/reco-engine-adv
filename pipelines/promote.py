"""Generation-filtered promotion implemented as a fenced compensating saga.

Vertex datapoints from old and new generations coexist. The only serving switch
is the BigQuery active-generation row; API queries filter that exact generation.
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
from collections.abc import Callable
from typing import Any

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import bigquery, storage


ROOT = pathlib.Path(__file__).resolve().parents[1]
for sibling in (ROOT / "training", ROOT / "featurestore"):
    if sibling.is_dir():
        sys.path.insert(0, str(sibling))

from generation_contract import (  # noqa: E402
    ACTIVE_GENERATION_TABLE,
    GENERATION_RETIREMENTS_TABLE,
    INITIALIZATION_LOCK_BLOB,
    USER_GENERATIONS_TABLE,
    ManifestReceipt,
    ServingBinding,
    active_state_receipt_sha256,
    canonical_user_payload,
    item_blob_name,
    manifest_blob_name,
    manifest_uri,
    parse_item_payload,
    parse_user_payload,
    sha256_bytes,
    user_blob_name,
    validate_binding,
    validate_generation_id,
    validate_manifest,
    validate_ops_image_uri,
    validate_sha256,
    validate_source_revision,
)
from refresh_index import (  # noqa: E402
    upsert_generation,
    verify_live_vertex_target,
    wait_for_generation_visibility,
)


PROJECT = os.environ["PROJECT"]
REGION = os.environ.get("REGION", "europe-west3")
BUCKET = os.environ.get("ARTIFACT_BUCKET", f"{PROJECT}-artifacts")
VECTOR_BUCKET = os.environ.get("VECTOR_BUCKET", f"{PROJECT}-vectors")
LEASE_TTL_SECONDS = max(30, min(int(os.environ.get("LEASE_TTL_SECONDS", "300")), 3600))
INITIALIZATION_COMMIT_TTL_SECONDS = max(
    60, min(int(os.environ.get("INITIALIZATION_COMMIT_TTL_SECONDS", "900")), 3600)
)
GENERATION_KEEP_COUNT = max(
    3, min(int(os.environ.get("GENERATION_KEEP_COUNT", "3")), 20)
)
GENERATION_DRAIN_SECONDS = max(
    300, min(int(os.environ.get("GENERATION_DRAIN_SECONDS", "86400")), 604800)
)


@dataclass(frozen=True)
class ActiveState:
    generation_id: str
    epoch: int
    previous_generation_id: str | None
    manifest_uri: str
    manifest_object_generation: int
    manifest_sha256: str
    ops_image_uri: str
    source_revision: str
    retained_generation_ids: tuple[str, ...]
    champion_score: float
    binding: ServingBinding
    promoted_at: Any
    lease_owner: str | None
    lease_epoch: int | None
    lease_phase: str | None
    lease_acquired_at: Any | None
    lease_heartbeat_at: Any | None
    lease_expires_at: Any | None
    pending_generation_id: str | None
    pending_manifest_uri: str | None
    pending_manifest_object_generation: int | None
    pending_manifest_sha256: str | None
    pending_ops_image_uri: str | None
    pending_source_revision: str | None
    pending_score: float | None


@dataclass(frozen=True)
class LeaseFence:
    owner: str
    lease_epoch: int
    active_generation: str
    active_epoch: int
    active_manifest_uri: str
    active_manifest_object_generation: int
    active_manifest_sha256: str
    active_ops_image_uri: str
    active_source_revision: str
    active_champion_score: float
    candidate_generation: str
    candidate_manifest_uri: str
    candidate_manifest_object_generation: int
    candidate_manifest_sha256: str
    candidate_ops_image_uri: str
    candidate_source_revision: str
    candidate_score: float
    binding: ServingBinding


@dataclass(frozen=True)
class InitializationCommitFence:
    owner: str
    lease_epoch: int
    generation_id: str
    object_generation: int
    expires_at: datetime
    binding: ServingBinding
    candidate_manifest_uri: str
    candidate_manifest_object_generation: int
    candidate_manifest_sha256: str
    candidate_ops_image_uri: str
    candidate_source_revision: str
    candidate_score: float


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _query_config(**parameters: tuple[str, Any]) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(name, field_type, value)
            for name, (field_type, value) in parameters.items()
        ]
    )


def conditional_download(
    blob,
    expected_generation: int | None = None,
    fence: Callable[[], None] | None = None,
) -> tuple[bytes, int]:
    if fence:
        fence()
    blob.reload()
    generation = int(blob.generation)
    if expected_generation is not None and generation != expected_generation:
        raise RuntimeError("immutable GCS object generation differs from its manifest")
    if fence:
        fence()
    return blob.download_as_bytes(if_generation_match=generation), generation


def _ensure_table_schema(
    bq: bigquery.Client, table_id: str, schema: list[bigquery.SchemaField]
) -> None:
    """Create once, or idempotently add nullable migration columns."""
    try:
        table = bq.get_table(table_id)
    except NotFound:
        bq.create_table(bigquery.Table(table_id, schema=schema))
        return
    existing = {field.name for field in table.schema}
    additions = []
    for field in schema:
        if field.name in existing:
            continue
        # BigQuery can add nullable/repeated fields, but not a REQUIRED field to
        # a populated table. Runtime validation still fails closed on NULL.
        mode = field.mode if field.mode == "REPEATED" else "NULLABLE"
        additions.append(bigquery.SchemaField(field.name, field.field_type, mode=mode))
    if additions:
        table.schema = list(table.schema) + additions
        bq.update_table(table, ["schema"])


def ensure_tables(bq: bigquery.Client) -> None:
    _ensure_table_schema(
        bq,
        f"{PROJECT}.reco.{USER_GENERATIONS_TABLE}",
        [
            bigquery.SchemaField("generation_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
            bigquery.SchemaField("trained_at", "TIMESTAMP", mode="REQUIRED"),
        ],
    )
    _ensure_table_schema(
        bq,
        f"{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}",
        [
            bigquery.SchemaField("singleton", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("generation_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("epoch", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("previous_generation_id", "STRING"),
            bigquery.SchemaField("manifest_uri", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "manifest_object_generation", "INTEGER", mode="REQUIRED"
            ),
            bigquery.SchemaField("manifest_sha256", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("ops_image_uri", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source_revision", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "retained_generation_ids", "STRING", mode="REPEATED"
            ),
            bigquery.SchemaField("champion_score", "FLOAT"),
            bigquery.SchemaField("index_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("index_endpoint", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("deployed_index_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("promoted_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("lease_owner", "STRING"),
            bigquery.SchemaField("lease_epoch", "INTEGER"),
            bigquery.SchemaField("lease_phase", "STRING"),
            bigquery.SchemaField("lease_acquired_at", "TIMESTAMP"),
            bigquery.SchemaField("lease_heartbeat_at", "TIMESTAMP"),
            bigquery.SchemaField("lease_expires_at", "TIMESTAMP"),
            bigquery.SchemaField("pending_generation_id", "STRING"),
            bigquery.SchemaField("pending_manifest_uri", "STRING"),
            bigquery.SchemaField("pending_manifest_object_generation", "INTEGER"),
            bigquery.SchemaField("pending_manifest_sha256", "STRING"),
            bigquery.SchemaField("pending_ops_image_uri", "STRING"),
            bigquery.SchemaField("pending_source_revision", "STRING"),
            bigquery.SchemaField("pending_score", "FLOAT"),
        ],
    )
    _ensure_table_schema(
        bq,
        f"{PROJECT}.reco.{GENERATION_RETIREMENTS_TABLE}",
        [
            bigquery.SchemaField("generation_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("retired_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("drain_until", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("retired_epoch", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField(
                "successor_generation_id", "STRING", mode="REQUIRED"
            ),
            bigquery.SchemaField("manifest_uri", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "manifest_object_generation", "INTEGER", mode="REQUIRED"
            ),
            bigquery.SchemaField("manifest_sha256", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("index_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("index_endpoint", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("deployed_index_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("ops_image_uri", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source_revision", "STRING", mode="REQUIRED"),
        ],
    )


def ensure_serving_view(bq: bigquery.Client) -> None:
    bq.query(
        f"""CREATE OR REPLACE VIEW `{PROJECT}.reco.user_embeddings_latest` AS
            SELECT u.user_id, u.embedding, u.trained_at
            FROM `{PROJECT}.reco.{USER_GENERATIONS_TABLE}` u
            JOIN `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}` a
              ON a.singleton = TRUE AND u.generation_id = a.generation_id
            QUALIFY ROW_NUMBER() OVER (
              PARTITION BY u.user_id ORDER BY u.trained_at DESC
            ) = 1"""
    ).result()


def _row_value(row: Any, field: str) -> Any:
    try:
        return row[field]
    except (KeyError, TypeError, IndexError):
        return None


def state_receipt_sha256(state: ActiveState) -> str:
    return active_state_receipt_sha256(
        generation_id=state.generation_id,
        epoch=state.epoch,
        manifest_uri_value=state.manifest_uri,
        manifest_object_generation=state.manifest_object_generation,
        manifest_sha256=state.manifest_sha256,
        ops_image_uri=state.ops_image_uri,
        source_revision=state.source_revision,
        champion_score=state.champion_score,
        binding=state.binding,
    )


def read_active_state(bq: bigquery.Client) -> ActiveState | None:
    try:
        bq.get_table(f"{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}")
    except NotFound:
        return None
    rows = list(
        bq.query(
            f"""SELECT * FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
                WHERE singleton = TRUE"""
        ).result()
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise RuntimeError("active generation table contains multiple singleton rows")
    row = rows[0]
    receipt_fields = (
        "manifest_object_generation",
        "manifest_sha256",
        "ops_image_uri",
        "source_revision",
    )
    if any(_row_value(row, field) is None for field in receipt_fields):
        raise RuntimeError(
            "legacy active state has no immutable receipt; explicit migration is required"
        )
    generation_id = validate_generation_id(str(row["generation_id"]))
    champion_score = _row_value(row, "champion_score")
    if champion_score is None:
        raise RuntimeError("active state has no champion score")
    state = ActiveState(
        generation_id=generation_id,
        epoch=int(row["epoch"]),
        previous_generation_id=row["previous_generation_id"],
        manifest_uri=str(row["manifest_uri"]),
        manifest_object_generation=int(row["manifest_object_generation"]),
        manifest_sha256=validate_sha256(str(row["manifest_sha256"])),
        ops_image_uri=validate_ops_image_uri(str(row["ops_image_uri"])),
        source_revision=validate_source_revision(str(row["source_revision"])),
        retained_generation_ids=tuple(_row_value(row, "retained_generation_ids") or ()),
        champion_score=float(champion_score),
        binding=validate_binding(
            row["index_name"], row["index_endpoint"], row["deployed_index_id"]
        ),
        promoted_at=row["promoted_at"],
        lease_owner=row["lease_owner"],
        lease_epoch=int(row["lease_epoch"]) if row["lease_epoch"] is not None else None,
        lease_phase=row["lease_phase"],
        lease_acquired_at=row["lease_acquired_at"],
        lease_heartbeat_at=row["lease_heartbeat_at"],
        lease_expires_at=row["lease_expires_at"],
        pending_generation_id=row["pending_generation_id"],
        pending_manifest_uri=row["pending_manifest_uri"],
        pending_manifest_object_generation=(
            int(row["pending_manifest_object_generation"])
            if _row_value(row, "pending_manifest_object_generation") is not None
            else None
        ),
        pending_manifest_sha256=_row_value(row, "pending_manifest_sha256"),
        pending_ops_image_uri=_row_value(row, "pending_ops_image_uri"),
        pending_source_revision=_row_value(row, "pending_source_revision"),
        pending_score=(
            float(row["pending_score"])
            if _row_value(row, "pending_score") is not None
            else None
        ),
    )
    if (
        state.epoch <= 0
        or state.manifest_object_generation <= 0
        or state.manifest_uri != manifest_uri(BUCKET, generation_id)
        or not 0.0 <= state.champion_score <= 1.0
    ):
        raise RuntimeError("active generation pointer is malformed")
    retained = tuple(validate_generation_id(str(value)) for value in state.retained_generation_ids)
    if len(set(retained)) != len(retained) or generation_id not in retained:
        raise RuntimeError("active generation retention state is malformed")
    if state.previous_generation_id is not None:
        previous = validate_generation_id(str(state.previous_generation_id))
        if previous not in retained:
            raise RuntimeError("rollback generation is absent from retained state")
    if state.lease_epoch is not None and state.lease_epoch < state.epoch:
        raise RuntimeError("active generation lease epoch moved backwards")
    lease_fields = (
        state.lease_phase,
        state.lease_acquired_at,
        state.lease_heartbeat_at,
        state.lease_expires_at,
        state.pending_generation_id,
        state.pending_manifest_uri,
        state.pending_manifest_object_generation,
        state.pending_manifest_sha256,
        state.pending_ops_image_uri,
        state.pending_source_revision,
        state.pending_score,
    )
    if state.lease_owner:
        if (
            state.lease_epoch is None
            or state.lease_epoch <= state.epoch
            or any(value is None for value in lease_fields)
        ):
            raise RuntimeError("active generation contains an incomplete lease")
        validate_generation_id(str(state.pending_generation_id))
        if state.pending_manifest_uri != manifest_uri(
            BUCKET, str(state.pending_generation_id)
        ):
            raise RuntimeError("active generation lease manifest is malformed")
        if state.pending_manifest_object_generation <= 0:
            raise RuntimeError("active generation lease manifest version is malformed")
        validate_sha256(str(state.pending_manifest_sha256), "pending manifest digest")
        validate_ops_image_uri(str(state.pending_ops_image_uri))
        validate_source_revision(str(state.pending_source_revision))
        if not 0.0 <= float(state.pending_score) <= 1.0:
            raise RuntimeError("active generation lease score is malformed")
    elif any(value is not None for value in lease_fields):
        raise RuntimeError("active generation contains orphaned lease state")
    state_receipt_sha256(state)
    return state


def load_candidate(
    gcs: storage.Client,
    generation_id: str,
    binding: ServingBinding,
    receipt: ManifestReceipt,
    ops_image_uri: str,
    source_revision: str,
    fence: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if receipt.uri != manifest_uri(BUCKET, generation_id):
        raise RuntimeError("candidate receipt names another manifest")
    if receipt.object_generation <= 0:
        raise RuntimeError("candidate manifest object generation is invalid")
    validate_sha256(receipt.sha256, "candidate manifest digest")
    ops_image_uri = validate_ops_image_uri(ops_image_uri)
    source_revision = validate_source_revision(source_revision)
    canonical = gcs.bucket(BUCKET)
    regional = gcs.bucket(VECTOR_BUCKET)
    manifest_payload, observed_generation = conditional_download(
        canonical.blob(manifest_blob_name(generation_id)),
        receipt.object_generation,
        fence,
    )
    if observed_generation != receipt.object_generation or sha256_bytes(manifest_payload) != receipt.sha256:
        raise RuntimeError("candidate manifest differs from its evaluation receipt")
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
        fence,
    )
    regional_items, _ = conditional_download(
        regional.blob(item_blob_name(generation_id)),
        manifest["items"]["regional_object_generation"],
        fence,
    )
    if (
        sha256_bytes(canonical_items) != manifest["items"]["sha256"]
        or sha256_bytes(regional_items) != manifest["items"]["sha256"]
    ):
        raise RuntimeError("candidate item artifact digest mismatch")
    items = parse_item_payload(canonical_items, dimensions=dimensions)
    if len(items) != manifest["items"]["count"]:
        raise RuntimeError("candidate item count mismatch")
    if manifest["items"]["canary_item_id"] not in {row["id"] for row in items}:
        raise RuntimeError("candidate item canary is absent from its immutable artifact")

    users_payload, _ = conditional_download(
        canonical.blob(user_blob_name(generation_id)),
        manifest["users"]["object_generation"],
        fence,
    )
    if sha256_bytes(users_payload) != manifest["users"]["sha256"]:
        raise RuntimeError("candidate user artifact digest mismatch")
    users = parse_user_payload(users_payload, dimensions=dimensions)
    if len(users) != manifest["users"]["count"]:
        raise RuntimeError("candidate user count mismatch")
    if manifest["users"]["canary_user_id"] not in {
        row["user_id"] for row in users
    }:
        raise RuntimeError("candidate user canary is absent from its immutable artifact")
    return manifest, items, users


def verify_staged_users(
    bq: bigquery.Client,
    manifest: dict[str, Any],
    generation_id: str,
    fence: Callable[[], None] | None = None,
) -> None:
    if fence:
        fence()
    rows = list(
        bq.query(
            f"""SELECT user_id, embedding
                FROM `{PROJECT}.reco.{USER_GENERATIONS_TABLE}`
                WHERE generation_id = @generation_id""",
            job_config=_query_config(generation_id=("STRING", generation_id)),
        ).result()
    )
    payload, normalized = canonical_user_payload(
        rows, dimensions=manifest["dimensions"]
    )
    if (
        len(normalized) != manifest["users"]["count"]
        or sha256_bytes(payload) != manifest["users"]["sha256"]
        or manifest["users"]["canary_user_id"]
        not in {row["user_id"] for row in normalized}
    ):
        raise RuntimeError("BigQuery staging differs from the immutable user generation")


def matching_resources(binding: ServingBinding):
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT, location=REGION)
    return (
        aiplatform.MatchingEngineIndex(index_name=binding.index_name),
        aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=binding.index_endpoint
        ),
    )


def _state_matches_fence(state: ActiveState | None, fence: LeaseFence) -> bool:
    return bool(
        state
        and state.generation_id == fence.active_generation
        and state.epoch == fence.active_epoch
        and state.manifest_uri == fence.active_manifest_uri
        and state.manifest_object_generation == fence.active_manifest_object_generation
        and state.manifest_sha256 == fence.active_manifest_sha256
        and state.ops_image_uri == fence.active_ops_image_uri
        and state.source_revision == fence.active_source_revision
        and state.champion_score == fence.active_champion_score
        and state.binding == fence.binding
        and state.lease_owner == fence.owner
        and state.lease_epoch == fence.lease_epoch
        and state.pending_generation_id == fence.candidate_generation
        and state.pending_manifest_uri == fence.candidate_manifest_uri
        and state.pending_manifest_object_generation
        == fence.candidate_manifest_object_generation
        and state.pending_manifest_sha256 == fence.candidate_manifest_sha256
        and state.pending_ops_image_uri == fence.candidate_ops_image_uri
        and state.pending_source_revision == fence.candidate_source_revision
        and state.pending_score == fence.candidate_score
        and state.lease_expires_at
        and state.lease_expires_at > now_utc()
    )


def fence_lease(bq: bigquery.Client, fence: LeaseFence) -> None:
    if not _state_matches_fence(read_active_state(bq), fence):
        raise RuntimeError("promotion lease fence was lost or expired")


def heartbeat_lease(bq: bigquery.Client, fence: LeaseFence, phase: str) -> None:
    sql = f"""
    BEGIN TRANSACTION;
    ASSERT (SELECT COUNT(*) FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      WHERE singleton = TRUE
        AND generation_id = @active_generation AND epoch = @active_epoch
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND lease_owner = @owner AND lease_epoch = @lease_epoch
        AND pending_generation_id = @candidate_generation
        AND pending_manifest_uri = @candidate_manifest_uri
        AND pending_manifest_object_generation = @candidate_manifest_object_generation
        AND pending_manifest_sha256 = @candidate_manifest_sha256
        AND pending_ops_image_uri = @candidate_ops_image_uri
        AND pending_source_revision = @candidate_source_revision
        AND pending_score = @candidate_score
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id
        AND lease_expires_at > CURRENT_TIMESTAMP()) = 1 AS 'promotion fence lost';
    UPDATE `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      SET lease_phase = @phase,
          lease_heartbeat_at = CURRENT_TIMESTAMP(),
          lease_expires_at = TIMESTAMP_ADD(
            CURRENT_TIMESTAMP(), INTERVAL {LEASE_TTL_SECONDS} SECOND)
      WHERE singleton = TRUE
        AND generation_id = @active_generation AND epoch = @active_epoch
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND lease_owner = @owner AND lease_epoch = @lease_epoch
        AND pending_generation_id = @candidate_generation
        AND pending_manifest_uri = @candidate_manifest_uri
        AND pending_manifest_object_generation = @candidate_manifest_object_generation
        AND pending_manifest_sha256 = @candidate_manifest_sha256
        AND pending_ops_image_uri = @candidate_ops_image_uri
        AND pending_source_revision = @candidate_source_revision
        AND pending_score = @candidate_score
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id
        AND lease_expires_at > CURRENT_TIMESTAMP();
    COMMIT TRANSACTION;
    """
    bq.query(
        sql,
        job_config=_query_config(
            active_generation=("STRING", fence.active_generation),
            active_epoch=("INT64", fence.active_epoch),
            active_manifest_uri=("STRING", fence.active_manifest_uri),
            active_manifest_object_generation=(
                "INT64",
                fence.active_manifest_object_generation,
            ),
            active_manifest_sha256=("STRING", fence.active_manifest_sha256),
            active_ops_image_uri=("STRING", fence.active_ops_image_uri),
            active_source_revision=("STRING", fence.active_source_revision),
            active_champion_score=("FLOAT64", fence.active_champion_score),
            owner=("STRING", fence.owner),
            lease_epoch=("INT64", fence.lease_epoch),
            candidate_generation=("STRING", fence.candidate_generation),
            candidate_manifest_uri=("STRING", fence.candidate_manifest_uri),
            candidate_manifest_object_generation=(
                "INT64",
                fence.candidate_manifest_object_generation,
            ),
            candidate_manifest_sha256=("STRING", fence.candidate_manifest_sha256),
            candidate_ops_image_uri=("STRING", fence.candidate_ops_image_uri),
            candidate_source_revision=("STRING", fence.candidate_source_revision),
            candidate_score=("FLOAT64", fence.candidate_score),
            index_name=("STRING", fence.binding.index_name),
            index_endpoint=("STRING", fence.binding.index_endpoint),
            deployed_index_id=("STRING", fence.binding.deployed_index_id),
            phase=("STRING", phase),
        ),
    ).result()
    fence_lease(bq, fence)


def release_lease(bq: bigquery.Client, fence: LeaseFence) -> None:
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
                AND lease_owner = @owner AND lease_epoch = @lease_epoch
                AND pending_generation_id = @candidate_generation
                AND pending_manifest_uri = @candidate_manifest_uri
                AND pending_manifest_object_generation = @candidate_manifest_object_generation
                AND pending_manifest_sha256 = @candidate_manifest_sha256
                AND pending_ops_image_uri = @candidate_ops_image_uri
                AND pending_source_revision = @candidate_source_revision
                AND pending_score = @candidate_score
                AND index_name = @index_name AND index_endpoint = @index_endpoint
                AND deployed_index_id = @deployed_index_id) = 1
              AS 'lease release fence lost';
            UPDATE `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
            SET lease_owner = NULL, lease_phase = NULL,
                lease_acquired_at = NULL, lease_heartbeat_at = NULL,
                lease_expires_at = NULL, pending_generation_id = NULL,
                pending_manifest_uri = NULL,
                pending_manifest_object_generation = NULL,
                pending_manifest_sha256 = NULL, pending_ops_image_uri = NULL,
                pending_source_revision = NULL, pending_score = NULL
            WHERE singleton = TRUE
              AND generation_id = @active_generation AND epoch = @active_epoch
              AND manifest_uri = @active_manifest_uri
              AND manifest_object_generation = @active_manifest_object_generation
              AND manifest_sha256 = @active_manifest_sha256
              AND ops_image_uri = @active_ops_image_uri
              AND source_revision = @active_source_revision
              AND champion_score = @active_champion_score
              AND lease_owner = @owner AND lease_epoch = @lease_epoch
              AND pending_generation_id = @candidate_generation
              AND pending_manifest_uri = @candidate_manifest_uri
              AND pending_manifest_object_generation = @candidate_manifest_object_generation
              AND pending_manifest_sha256 = @candidate_manifest_sha256
              AND pending_ops_image_uri = @candidate_ops_image_uri
              AND pending_source_revision = @candidate_source_revision
              AND pending_score = @candidate_score
              AND index_name = @index_name AND index_endpoint = @index_endpoint
              AND deployed_index_id = @deployed_index_id;
            COMMIT TRANSACTION;""",
        job_config=_query_config(
            active_generation=("STRING", fence.active_generation),
            active_epoch=("INT64", fence.active_epoch),
            active_manifest_uri=("STRING", fence.active_manifest_uri),
            active_manifest_object_generation=(
                "INT64",
                fence.active_manifest_object_generation,
            ),
            active_manifest_sha256=("STRING", fence.active_manifest_sha256),
            active_ops_image_uri=("STRING", fence.active_ops_image_uri),
            active_source_revision=("STRING", fence.active_source_revision),
            active_champion_score=("FLOAT64", fence.active_champion_score),
            owner=("STRING", fence.owner),
            lease_epoch=("INT64", fence.lease_epoch),
            candidate_generation=("STRING", fence.candidate_generation),
            candidate_manifest_uri=("STRING", fence.candidate_manifest_uri),
            candidate_manifest_object_generation=(
                "INT64",
                fence.candidate_manifest_object_generation,
            ),
            candidate_manifest_sha256=("STRING", fence.candidate_manifest_sha256),
            candidate_ops_image_uri=("STRING", fence.candidate_ops_image_uri),
            candidate_source_revision=("STRING", fence.candidate_source_revision),
            candidate_score=("FLOAT64", fence.candidate_score),
            index_name=("STRING", fence.binding.index_name),
            index_endpoint=("STRING", fence.binding.index_endpoint),
            deployed_index_id=("STRING", fence.binding.deployed_index_id),
        ),
    ).result()


def recover_stale_lease(
    bq: bigquery.Client, state: ActiveState, recovery_owner: str
) -> None:
    if not state.lease_owner or not state.lease_epoch or not state.lease_expires_at:
        raise RuntimeError("cannot recover an incomplete lease record")
    if state.lease_expires_at > now_utc():
        raise RuntimeError("another promotion owns a non-stale lease")
    if not state.pending_generation_id or not state.pending_manifest_uri:
        raise RuntimeError("stale lease has no fenced candidate to recover")
    sql = f"""
    BEGIN TRANSACTION;
    ASSERT (SELECT COUNT(*) FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      WHERE singleton = TRUE AND generation_id = @active_generation
        AND epoch = @active_epoch AND lease_owner = @old_owner
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND lease_epoch = @lease_epoch
        AND pending_generation_id = @candidate_generation
        AND pending_manifest_uri = @candidate_manifest_uri
        AND pending_manifest_object_generation = @candidate_manifest_object_generation
        AND pending_manifest_sha256 = @candidate_manifest_sha256
        AND pending_ops_image_uri = @candidate_ops_image_uri
        AND pending_source_revision = @candidate_source_revision
        AND pending_score = @candidate_score
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id
        AND lease_expires_at <= CURRENT_TIMESTAMP()) = 1
      AS 'stale lease changed before recovery';
    UPDATE `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      SET lease_owner = @recovery_owner, lease_phase = 'recovering',
          lease_acquired_at = CURRENT_TIMESTAMP(),
          lease_heartbeat_at = CURRENT_TIMESTAMP(),
          lease_expires_at = TIMESTAMP_ADD(
            CURRENT_TIMESTAMP(), INTERVAL {LEASE_TTL_SECONDS} SECOND)
      WHERE singleton = TRUE AND generation_id = @active_generation
        AND epoch = @active_epoch AND lease_owner = @old_owner
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND lease_epoch = @lease_epoch
        AND pending_generation_id = @candidate_generation
        AND pending_manifest_uri = @candidate_manifest_uri
        AND pending_manifest_object_generation = @candidate_manifest_object_generation
        AND pending_manifest_sha256 = @candidate_manifest_sha256
        AND pending_ops_image_uri = @candidate_ops_image_uri
        AND pending_source_revision = @candidate_source_revision
        AND pending_score = @candidate_score
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id
        AND lease_expires_at <= CURRENT_TIMESTAMP();
    COMMIT TRANSACTION;
    """
    bq.query(
        sql,
        job_config=_query_config(
            active_generation=("STRING", state.generation_id),
            active_epoch=("INT64", state.epoch),
            active_manifest_uri=("STRING", state.manifest_uri),
            active_manifest_object_generation=(
                "INT64",
                state.manifest_object_generation,
            ),
            active_manifest_sha256=("STRING", state.manifest_sha256),
            active_ops_image_uri=("STRING", state.ops_image_uri),
            active_source_revision=("STRING", state.source_revision),
            active_champion_score=("FLOAT64", state.champion_score),
            old_owner=("STRING", state.lease_owner),
            lease_epoch=("INT64", state.lease_epoch),
            recovery_owner=("STRING", recovery_owner),
            candidate_generation=("STRING", state.pending_generation_id),
            candidate_manifest_uri=("STRING", state.pending_manifest_uri),
            candidate_manifest_object_generation=(
                "INT64",
                state.pending_manifest_object_generation,
            ),
            candidate_manifest_sha256=("STRING", state.pending_manifest_sha256),
            candidate_ops_image_uri=("STRING", state.pending_ops_image_uri),
            candidate_source_revision=("STRING", state.pending_source_revision),
            candidate_score=("FLOAT64", state.pending_score),
            index_name=("STRING", state.binding.index_name),
            index_endpoint=("STRING", state.binding.index_endpoint),
            deployed_index_id=("STRING", state.binding.deployed_index_id),
        ),
    ).result()
    recovered = read_active_state(bq)
    if not recovered or recovered.lease_owner != recovery_owner:
        raise RuntimeError("stale lease recovery fence was not acquired")
    recovery_fence = LeaseFence(
        owner=recovery_owner,
        lease_epoch=state.lease_epoch,
        active_generation=state.generation_id,
        active_epoch=state.epoch,
        active_manifest_uri=state.manifest_uri,
        active_manifest_object_generation=state.manifest_object_generation,
        active_manifest_sha256=state.manifest_sha256,
        active_ops_image_uri=state.ops_image_uri,
        active_source_revision=state.source_revision,
        active_champion_score=state.champion_score,
        candidate_generation=state.pending_generation_id,
        candidate_manifest_uri=state.pending_manifest_uri,
        candidate_manifest_object_generation=state.pending_manifest_object_generation,
        candidate_manifest_sha256=state.pending_manifest_sha256,
        candidate_ops_image_uri=state.pending_ops_image_uri,
        candidate_source_revision=state.pending_source_revision,
        candidate_score=state.pending_score,
        binding=state.binding,
    )
    fence_lease(bq, recovery_fence)
    release_lease(bq, recovery_fence)


def acquire_lease(
    bq: bigquery.Client,
    *,
    state: ActiveState,
    owner: str,
    candidate_generation: str,
    candidate_receipt: ManifestReceipt,
    candidate_ops_image_uri: str,
    candidate_source_revision: str,
    candidate_score: float,
    evaluated_active_receipt_sha256: str,
) -> LeaseFence:
    if candidate_receipt.uri != manifest_uri(BUCKET, candidate_generation):
        raise RuntimeError("candidate lease receipt names another manifest")
    if candidate_receipt.object_generation <= 0:
        raise RuntimeError("candidate lease manifest generation is invalid")
    validate_sha256(candidate_receipt.sha256, "candidate lease manifest digest")
    candidate_ops_image_uri = validate_ops_image_uri(candidate_ops_image_uri)
    candidate_source_revision = validate_source_revision(candidate_source_revision)
    if not 0.0 <= candidate_score <= 1.0:
        raise ValueError("candidate lease score must be in [0, 1]")
    validate_sha256(evaluated_active_receipt_sha256, "evaluated active receipt")
    current = read_active_state(bq)
    if current != state:
        raise RuntimeError("active state changed after evaluation")
    if state_receipt_sha256(current) != evaluated_active_receipt_sha256:
        raise RuntimeError("active immutable receipt changed after evaluation")
    if current.lease_owner:
        if current.lease_expires_at and current.lease_expires_at > now_utc():
            raise RuntimeError("another promotion owns a non-stale lease")
        recover_stale_lease(bq, current, f"recovery-{uuid.uuid4()}")
        current = read_active_state(bq)
        if current is None or current.lease_owner:
            raise RuntimeError("stale promotion lease was not safely cleared")
        if state_receipt_sha256(current) != evaluated_active_receipt_sha256:
            raise RuntimeError("active receipt changed during stale lease recovery")
    prior_fence_epoch = max(current.epoch, current.lease_epoch or 0)
    lease_epoch = prior_fence_epoch + 1
    sql = f"""
    BEGIN TRANSACTION;
    ASSERT (SELECT COUNT(*) FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      WHERE singleton = TRUE AND generation_id = @active_generation
        AND epoch = @active_epoch AND lease_owner IS NULL
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND COALESCE(lease_epoch, epoch) = @prior_fence_epoch
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id) = 1 AS 'active fence changed';
    UPDATE `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      SET lease_owner = @owner, lease_epoch = @lease_epoch, lease_phase = 'acquired',
          lease_acquired_at = CURRENT_TIMESTAMP(),
          lease_heartbeat_at = CURRENT_TIMESTAMP(),
          lease_expires_at = TIMESTAMP_ADD(
            CURRENT_TIMESTAMP(), INTERVAL {LEASE_TTL_SECONDS} SECOND),
          pending_generation_id = @candidate_generation,
          pending_manifest_uri = @candidate_manifest_uri,
          pending_manifest_object_generation = @candidate_manifest_object_generation,
          pending_manifest_sha256 = @candidate_manifest_sha256,
          pending_ops_image_uri = @candidate_ops_image_uri,
          pending_source_revision = @candidate_source_revision,
          pending_score = @candidate_score
      WHERE singleton = TRUE AND generation_id = @active_generation
        AND epoch = @active_epoch AND lease_owner IS NULL
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND COALESCE(lease_epoch, epoch) = @prior_fence_epoch
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id;
    COMMIT TRANSACTION;
    """
    bq.query(
        sql,
        job_config=_query_config(
            active_generation=("STRING", current.generation_id),
            active_epoch=("INT64", current.epoch),
            active_manifest_uri=("STRING", current.manifest_uri),
            active_manifest_object_generation=(
                "INT64",
                current.manifest_object_generation,
            ),
            active_manifest_sha256=("STRING", current.manifest_sha256),
            active_ops_image_uri=("STRING", current.ops_image_uri),
            active_source_revision=("STRING", current.source_revision),
            active_champion_score=("FLOAT64", current.champion_score),
            prior_fence_epoch=("INT64", prior_fence_epoch),
            index_name=("STRING", current.binding.index_name),
            index_endpoint=("STRING", current.binding.index_endpoint),
            deployed_index_id=("STRING", current.binding.deployed_index_id),
            owner=("STRING", owner),
            lease_epoch=("INT64", lease_epoch),
            candidate_generation=("STRING", candidate_generation),
            candidate_manifest_uri=("STRING", candidate_receipt.uri),
            candidate_manifest_object_generation=(
                "INT64",
                candidate_receipt.object_generation,
            ),
            candidate_manifest_sha256=("STRING", candidate_receipt.sha256),
            candidate_ops_image_uri=("STRING", candidate_ops_image_uri),
            candidate_source_revision=("STRING", candidate_source_revision),
            candidate_score=("FLOAT64", candidate_score),
        ),
    ).result()
    fence = LeaseFence(
        owner=owner,
        lease_epoch=lease_epoch,
        active_generation=current.generation_id,
        active_epoch=current.epoch,
        active_manifest_uri=current.manifest_uri,
        active_manifest_object_generation=current.manifest_object_generation,
        active_manifest_sha256=current.manifest_sha256,
        active_ops_image_uri=current.ops_image_uri,
        active_source_revision=current.source_revision,
        active_champion_score=current.champion_score,
        candidate_generation=candidate_generation,
        candidate_manifest_uri=candidate_receipt.uri,
        candidate_manifest_object_generation=candidate_receipt.object_generation,
        candidate_manifest_sha256=candidate_receipt.sha256,
        candidate_ops_image_uri=candidate_ops_image_uri,
        candidate_source_revision=candidate_source_revision,
        candidate_score=candidate_score,
        binding=current.binding,
    )
    fence_lease(bq, fence)
    return fence


def _initial_lock_payload(
    *,
    owner: str,
    lease_epoch: int,
    phase: str,
    generation_id: str,
    binding: ServingBinding,
    receipt: ManifestReceipt,
    ops_image_uri: str,
    source_revision: str,
    score: float,
    ttl_seconds: int = LEASE_TTL_SECONDS,
) -> dict[str, Any]:
    now = now_utc()
    return {
        "schema_version": 2,
        "owner": owner,
        "lease_epoch": lease_epoch,
        "phase": phase,
        "generation_id": generation_id,
        "manifest_uri": receipt.uri,
        "manifest_object_generation": receipt.object_generation,
        "manifest_sha256": receipt.sha256,
        "ops_image_uri": ops_image_uri,
        "source_revision": source_revision,
        "score": score,
        "index_name": binding.index_name,
        "index_endpoint": binding.index_endpoint,
        "deployed_index_id": binding.deployed_index_id,
        "acquired_at": now.isoformat(),
        "heartbeat_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }


def _read_initial_lock(
    blob, expected_generation: int | None = None
) -> tuple[dict[str, Any], int]:
    payload, object_generation = conditional_download(blob, expected_generation)
    return json.loads(payload), object_generation


def _initial_fence_from_payload(
    payload: dict[str, Any], object_generation: int
) -> InitializationCommitFence:
    if payload.get("schema_version") != 2:
        raise RuntimeError("legacy initialization lock cannot be trusted")
    receipt = ManifestReceipt(
        uri=str(payload.get("manifest_uri")),
        object_generation=int(payload.get("manifest_object_generation")),
        sha256=validate_sha256(str(payload.get("manifest_sha256"))),
    )
    generation_id = validate_generation_id(str(payload.get("generation_id")))
    if receipt.uri != manifest_uri(BUCKET, generation_id):
        raise RuntimeError("initialization lock manifest is malformed")
    score = float(payload.get("score"))
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("initialization lock score is malformed")
    return InitializationCommitFence(
        owner=str(payload.get("owner")),
        lease_epoch=int(payload.get("lease_epoch")),
        generation_id=generation_id,
        object_generation=object_generation,
        expires_at=parse_timestamp(str(payload.get("expires_at"))),
        binding=validate_binding(
            payload.get("index_name"),
            payload.get("index_endpoint"),
            payload.get("deployed_index_id"),
        ),
        candidate_manifest_uri=receipt.uri,
        candidate_manifest_object_generation=receipt.object_generation,
        candidate_manifest_sha256=receipt.sha256,
        candidate_ops_image_uri=validate_ops_image_uri(
            str(payload.get("ops_image_uri"))
        ),
        candidate_source_revision=validate_source_revision(
            str(payload.get("source_revision"))
        ),
        candidate_score=score,
    )


def _assert_initialization_fence(
    gcs: storage.Client,
    fence: InitializationCommitFence,
    *,
    required_phase: str,
) -> dict[str, Any]:
    blob = gcs.bucket(BUCKET).blob(INITIALIZATION_LOCK_BLOB)
    payload, observed_generation = _read_initial_lock(blob, fence.object_generation)
    observed = _initial_fence_from_payload(payload, observed_generation)
    if observed != fence or payload.get("phase") != required_phase:
        raise RuntimeError("initialization lock owner/CAS fence was lost")
    if observed.expires_at <= now_utc():
        raise RuntimeError("initialization lock expired")
    return payload


def acquire_initialization_lock(
    bq: bigquery.Client,
    gcs: storage.Client,
    *,
    owner: str,
    generation_id: str,
    binding: ServingBinding,
    receipt: ManifestReceipt,
    ops_image_uri: str,
    source_revision: str,
    score: float,
) -> InitializationCommitFence:
    generation_id = validate_generation_id(generation_id)
    binding = validate_binding(
        binding.index_name, binding.index_endpoint, binding.deployed_index_id
    )
    if receipt.uri != manifest_uri(BUCKET, generation_id):
        raise RuntimeError("initialization receipt names another manifest")
    if receipt.object_generation <= 0:
        raise RuntimeError("initialization manifest generation is invalid")
    validate_sha256(receipt.sha256, "initialization manifest digest")
    ops_image_uri = validate_ops_image_uri(ops_image_uri)
    source_revision = validate_source_revision(source_revision)
    if not 0.0 <= score <= 1.0:
        raise ValueError("initialization score must be in [0, 1]")
    if read_active_state(bq) is not None:
        raise RuntimeError("active state appeared before initialization lock acquisition")
    blob = gcs.bucket(BUCKET).blob(INITIALIZATION_LOCK_BLOB)
    desired = _initial_lock_payload(
        owner=owner,
        lease_epoch=1,
        phase="acquired",
        generation_id=generation_id,
        binding=binding,
        receipt=receipt,
        ops_image_uri=ops_image_uri,
        source_revision=source_revision,
        score=score,
    )
    try:
        blob.upload_from_string(
            json.dumps(desired, sort_keys=True),
            content_type="application/json",
            if_generation_match=0,
        )
    except PreconditionFailed:
        existing, object_generation = _read_initial_lock(blob)
        if existing.get("phase") in {"committing", "complete"}:
            raise RuntimeError(
                "initialization commit is non-stealable; reconcile its active state"
            )
        existing_fence = _initial_fence_from_payload(existing, object_generation)
        if existing_fence.expires_at > now_utc():
            raise RuntimeError("another initializer owns a non-stale lock")
        if read_active_state(bq) is not None:
            raise RuntimeError("active state appeared before stale initialization recovery")
        desired = _initial_lock_payload(
            owner=owner,
            lease_epoch=existing_fence.lease_epoch + 1,
            phase="recovering",
            generation_id=generation_id,
            binding=binding,
            receipt=receipt,
            ops_image_uri=ops_image_uri,
            source_revision=source_revision,
            score=score,
        )
        blob.upload_from_string(
            json.dumps(desired, sort_keys=True),
            content_type="application/json",
            if_generation_match=object_generation,
        )
    blob.reload()
    payload, object_generation = _read_initial_lock(blob, int(blob.generation))
    fence = _initial_fence_from_payload(payload, object_generation)
    if fence.owner != owner or fence.generation_id != generation_id:
        raise RuntimeError("initialization lock CAS was stolen")
    if read_active_state(bq) is not None:
        raise RuntimeError("active state appeared during initialization lock acquisition")
    return fence


def heartbeat_initialization(
    bq: bigquery.Client,
    gcs: storage.Client,
    *,
    fence: InitializationCommitFence,
    phase: str,
    ttl_seconds: int = LEASE_TTL_SECONDS,
) -> InitializationCommitFence:
    if read_active_state(bq) is not None:
        raise RuntimeError("active state appeared during initialization")
    blob = gcs.bucket(BUCKET).blob(INITIALIZATION_LOCK_BLOB)
    current = _assert_initialization_fence(
        gcs, fence, required_phase=current_phase(gcs, fence)
    )
    current["phase"] = phase
    current["heartbeat_at"] = now_utc().isoformat()
    current["expires_at"] = (
        now_utc() + timedelta(seconds=ttl_seconds)
    ).isoformat()
    blob.upload_from_string(
        json.dumps(current, sort_keys=True),
        content_type="application/json",
        if_generation_match=fence.object_generation,
    )
    blob.reload()
    confirmed, observed_generation = _read_initial_lock(blob, int(blob.generation))
    updated = _initial_fence_from_payload(confirmed, observed_generation)
    if updated.owner != fence.owner or updated.lease_epoch != fence.lease_epoch:
        raise RuntimeError("initialization heartbeat CAS was stolen")
    if confirmed.get("phase") != phase or updated.expires_at <= now_utc():
        raise RuntimeError("initialization heartbeat was not fenced")
    if read_active_state(bq) is not None:
        raise RuntimeError("active state appeared after initialization lock CAS")
    return updated


def current_phase(gcs: storage.Client, fence: InitializationCommitFence) -> str:
    blob = gcs.bucket(BUCKET).blob(INITIALIZATION_LOCK_BLOB)
    payload, _ = _read_initial_lock(blob, fence.object_generation)
    phase = payload.get("phase")
    if not isinstance(phase, str) or not phase:
        raise RuntimeError("initialization lock phase is malformed")
    return phase


def mark_initialization_complete(
    gcs: storage.Client, *, fence: InitializationCommitFence
) -> InitializationCommitFence:
    blob = gcs.bucket(BUCKET).blob(INITIALIZATION_LOCK_BLOB)
    current = _assert_initialization_fence(gcs, fence, required_phase="committing")
    current["phase"] = "complete"
    current["heartbeat_at"] = now_utc().isoformat()
    current["expires_at"] = current["heartbeat_at"]
    blob.upload_from_string(
        json.dumps(current, sort_keys=True),
        content_type="application/json",
        if_generation_match=fence.object_generation,
    )
    blob.reload()
    payload, object_generation = _read_initial_lock(blob, int(blob.generation))
    if payload.get("phase") != "complete":
        raise RuntimeError("initialization completion CAS was not durable")
    return _initial_fence_from_payload(payload, object_generation)


def initialize_generation(
    *,
    bq: bigquery.Client,
    gcs: storage.Client,
    binding: ServingBinding,
    generation_id: str,
    manifest: dict[str, Any],
    item_rows: list[dict[str, Any]],
    receipt: ManifestReceipt,
    ops_image_uri: str,
    source_revision: str,
    score: float,
    owner: str,
) -> None:
    lock_fence = acquire_initialization_lock(
        bq,
        gcs,
        owner=owner,
        generation_id=generation_id,
        binding=binding,
        receipt=receipt,
        ops_image_uri=ops_image_uri,
        source_revision=source_revision,
        score=score,
    )

    def heartbeat(phase: str, ttl_seconds: int = LEASE_TTL_SECONDS) -> None:
        nonlocal lock_fence
        lock_fence = heartbeat_initialization(
            bq,
            gcs,
            fence=lock_fence,
            phase=phase,
            ttl_seconds=ttl_seconds,
        )

    heartbeat("preparing")
    ensure_serving_view(bq)
    heartbeat("binding")
    index, endpoint = matching_resources(binding)
    verify_live_vertex_target(
        index,
        endpoint,
        expected_index_name=binding.index_name,
        expected_endpoint_name=binding.index_endpoint,
        deployed_index_id=binding.deployed_index_id,
        expected_dimensions=manifest["dimensions"],
        fence=lambda: heartbeat("binding"),
    )
    heartbeat("upserting")
    specs = upsert_generation(
        index,
        rows=item_rows,
        generation_id=generation_id,
        fence=lambda: heartbeat("upserting"),
    )
    heartbeat("propagating")
    wait_for_generation_visibility(
        endpoint,
        deployed_index_id=binding.deployed_index_id,
        specs=specs,
        generation_id=generation_id,
        canary_datapoint_id=manifest["items"]["canary_datapoint_id"],
        fence=lambda: heartbeat("propagating"),
    )
    refreshed_manifest, _, _ = load_candidate(
        gcs,
        generation_id,
        binding,
        receipt,
        ops_image_uri,
        source_revision,
        fence=lambda: heartbeat("verifying"),
    )
    if refreshed_manifest != manifest:
        raise RuntimeError("candidate manifest changed during initialization")
    verify_staged_users(
        bq,
        manifest,
        generation_id,
        fence=lambda: heartbeat("verifying"),
    )
    verify_live_vertex_target(
        index,
        endpoint,
        expected_index_name=binding.index_name,
        expected_endpoint_name=binding.index_endpoint,
        deployed_index_id=binding.deployed_index_id,
        expected_dimensions=manifest["dimensions"],
        fence=lambda: heartbeat("final-live-binding"),
    )
    heartbeat("committing", INITIALIZATION_COMMIT_TTL_SECONDS)
    _assert_initialization_fence(gcs, lock_fence, required_phase="committing")
    if read_active_state(bq) is not None:
        raise RuntimeError("active generation was initialized concurrently")
    sql = f"""
    BEGIN TRANSACTION;
    INSERT INTO `{PROJECT}.reco.user_embeddings` (user_id, embedding, trained_at)
      SELECT user_id, embedding, trained_at
      FROM `{PROJECT}.reco.{USER_GENERATIONS_TABLE}`
      WHERE generation_id = @generation_id;
    MERGE `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}` target
    USING (SELECT TRUE AS singleton) source
      ON target.singleton = source.singleton
    WHEN NOT MATCHED THEN
      INSERT (singleton, generation_id, epoch, previous_generation_id, manifest_uri,
              manifest_object_generation, manifest_sha256, ops_image_uri,
              source_revision, retained_generation_ids, champion_score,
              index_name, index_endpoint, deployed_index_id, promoted_at, lease_epoch)
      VALUES (TRUE, @generation_id, 1, NULL, @manifest_uri,
              @manifest_object_generation, @manifest_sha256, @ops_image_uri,
              @source_revision, [@generation_id], @score,
              @index_name, @index_endpoint, @deployed_index_id,
              CURRENT_TIMESTAMP(), @initialization_lease_epoch);
    IF (SELECT COUNT(*) FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
        WHERE singleton = TRUE AND generation_id = @generation_id AND epoch = 1
          AND manifest_uri = @manifest_uri
          AND manifest_object_generation = @manifest_object_generation
          AND manifest_sha256 = @manifest_sha256
          AND ops_image_uri = @ops_image_uri AND source_revision = @source_revision
          AND champion_score = @score AND index_name = @index_name
          AND index_endpoint = @index_endpoint
          AND deployed_index_id = @deployed_index_id) != 1 THEN
      RAISE USING MESSAGE = 'initial active-generation seed was fenced out';
    END IF;
    COMMIT TRANSACTION;
    """
    commit_error: Exception | None = None
    try:
        bq.query(
            sql,
            job_config=_query_config(
                generation_id=("STRING", generation_id),
                manifest_uri=("STRING", receipt.uri),
                manifest_object_generation=("INT64", receipt.object_generation),
                manifest_sha256=("STRING", receipt.sha256),
                ops_image_uri=("STRING", ops_image_uri),
                source_revision=("STRING", source_revision),
                score=("FLOAT64", score),
                initialization_lease_epoch=("INT64", lock_fence.lease_epoch),
                index_name=("STRING", binding.index_name),
                index_endpoint=("STRING", binding.index_endpoint),
                deployed_index_id=("STRING", binding.deployed_index_id),
            ),
        ).result()
    except Exception as error:
        commit_error = error
    _assert_initialization_fence(gcs, lock_fence, required_phase="committing")
    state = read_active_state(bq)
    committed = bool(
        state
        and state.generation_id == generation_id
        and state.epoch == 1
        and state.binding == binding
        and state.manifest_uri == receipt.uri
        and state.manifest_object_generation == receipt.object_generation
        and state.manifest_sha256 == receipt.sha256
        and state.ops_image_uri == ops_image_uri
        and state.source_revision == source_revision
        and state.champion_score == score
    )
    if not committed:
        if commit_error is not None:
            raise commit_error
        raise RuntimeError("initial active-generation commit was not durable")
    mark_initialization_complete(gcs, fence=lock_fence)


def finalize_promotion(bq: bigquery.Client, fence: LeaseFence) -> None:
    sql = f"""
    BEGIN TRANSACTION;
    ASSERT (SELECT COUNT(*) FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      WHERE singleton = TRUE AND generation_id = @active_generation
        AND epoch = @active_epoch AND lease_owner = @owner
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND lease_epoch = @lease_epoch AND lease_phase = 'finalizing'
        AND lease_expires_at > CURRENT_TIMESTAMP()
        AND pending_generation_id = @candidate_generation
        AND pending_manifest_uri = @candidate_manifest_uri
        AND pending_manifest_object_generation = @candidate_manifest_object_generation
        AND pending_manifest_sha256 = @candidate_manifest_sha256
        AND pending_ops_image_uri = @candidate_ops_image_uri
        AND pending_source_revision = @candidate_source_revision
        AND pending_score = @candidate_score
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id) = 1 AS 'final promotion fence lost';
    INSERT INTO `{PROJECT}.reco.user_embeddings` (user_id, embedding, trained_at)
      SELECT user_id, embedding, trained_at
      FROM `{PROJECT}.reco.{USER_GENERATIONS_TABLE}`
      WHERE generation_id = @candidate_generation;
    UPDATE `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
      SET previous_generation_id = @active_generation,
          generation_id = @candidate_generation, epoch = @lease_epoch,
          manifest_uri = @candidate_manifest_uri,
          manifest_object_generation = @candidate_manifest_object_generation,
          manifest_sha256 = @candidate_manifest_sha256,
          ops_image_uri = @candidate_ops_image_uri,
          source_revision = @candidate_source_revision,
          retained_generation_ids = ARRAY(
            SELECT DISTINCT generation
            FROM UNNEST(ARRAY_CONCAT(
              IFNULL(retained_generation_ids, []),
              [@active_generation, @candidate_generation]
            )) AS generation
          ),
          champion_score = @candidate_score,
          promoted_at = CURRENT_TIMESTAMP(), lease_owner = NULL,
          lease_phase = NULL, lease_acquired_at = NULL, lease_heartbeat_at = NULL,
          lease_expires_at = NULL, pending_generation_id = NULL,
          pending_manifest_uri = NULL,
          pending_manifest_object_generation = NULL,
          pending_manifest_sha256 = NULL, pending_ops_image_uri = NULL,
          pending_source_revision = NULL, pending_score = NULL
      WHERE singleton = TRUE AND generation_id = @active_generation
        AND epoch = @active_epoch AND lease_owner = @owner
        AND manifest_uri = @active_manifest_uri
        AND manifest_object_generation = @active_manifest_object_generation
        AND manifest_sha256 = @active_manifest_sha256
        AND ops_image_uri = @active_ops_image_uri
        AND source_revision = @active_source_revision
        AND champion_score = @active_champion_score
        AND lease_epoch = @lease_epoch AND lease_phase = 'finalizing'
        AND lease_expires_at > CURRENT_TIMESTAMP()
        AND pending_generation_id = @candidate_generation
        AND pending_manifest_uri = @candidate_manifest_uri
        AND pending_manifest_object_generation = @candidate_manifest_object_generation
        AND pending_manifest_sha256 = @candidate_manifest_sha256
        AND pending_ops_image_uri = @candidate_ops_image_uri
        AND pending_source_revision = @candidate_source_revision
        AND pending_score = @candidate_score
        AND index_name = @index_name AND index_endpoint = @index_endpoint
        AND deployed_index_id = @deployed_index_id;
    COMMIT TRANSACTION;
    """
    bq.query(
        sql,
        job_config=_query_config(
            active_generation=("STRING", fence.active_generation),
            active_epoch=("INT64", fence.active_epoch),
            active_manifest_uri=("STRING", fence.active_manifest_uri),
            active_manifest_object_generation=(
                "INT64",
                fence.active_manifest_object_generation,
            ),
            active_manifest_sha256=("STRING", fence.active_manifest_sha256),
            active_ops_image_uri=("STRING", fence.active_ops_image_uri),
            active_source_revision=("STRING", fence.active_source_revision),
            active_champion_score=("FLOAT64", fence.active_champion_score),
            owner=("STRING", fence.owner),
            lease_epoch=("INT64", fence.lease_epoch),
            candidate_generation=("STRING", fence.candidate_generation),
            candidate_manifest_uri=("STRING", fence.candidate_manifest_uri),
            candidate_manifest_object_generation=(
                "INT64",
                fence.candidate_manifest_object_generation,
            ),
            candidate_manifest_sha256=("STRING", fence.candidate_manifest_sha256),
            candidate_ops_image_uri=("STRING", fence.candidate_ops_image_uri),
            candidate_source_revision=("STRING", fence.candidate_source_revision),
            candidate_score=("FLOAT64", fence.candidate_score),
            index_name=("STRING", fence.binding.index_name),
            index_endpoint=("STRING", fence.binding.index_endpoint),
            deployed_index_id=("STRING", fence.binding.deployed_index_id),
        ),
    ).result()


def promote(
    *,
    generation_id: str,
    score: float,
    evaluated_active_generation: str,
    evaluated_active_epoch: int,
    evaluated_manifest_generation: int,
    evaluated_manifest_sha256: str,
    evaluated_ops_image_uri: str,
    evaluated_source_revision: str,
    evaluated_active_manifest_generation: int,
    evaluated_active_manifest_sha256: str,
    evaluated_active_ops_image_uri: str,
    evaluated_active_source_revision: str,
    evaluated_active_champion_score: float,
    binding: ServingBinding,
) -> None:
    generation_id = validate_generation_id(generation_id)
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    if evaluated_manifest_generation <= 0:
        raise ValueError("evaluated candidate manifest generation must be positive")
    candidate_receipt = ManifestReceipt(
        uri=manifest_uri(BUCKET, generation_id),
        object_generation=evaluated_manifest_generation,
        sha256=validate_sha256(
            evaluated_manifest_sha256, "evaluated candidate manifest digest"
        ),
    )
    evaluated_ops_image_uri = validate_ops_image_uri(evaluated_ops_image_uri)
    evaluated_source_revision = validate_source_revision(evaluated_source_revision)
    owner = str(uuid.uuid4())
    bq = bigquery.Client(project=PROJECT)
    gcs = storage.Client(project=PROJECT)
    ensure_tables(bq)
    ensure_serving_view(bq)
    manifest, item_rows, _ = load_candidate(
        gcs,
        generation_id,
        binding,
        candidate_receipt,
        evaluated_ops_image_uri,
        evaluated_source_revision,
    )
    verify_staged_users(bq, manifest, generation_id)
    state = read_active_state(bq)

    if state and state.generation_id == generation_id:
        if not (
            state.binding == binding
            and state.champion_score == score
            and state.manifest_uri == candidate_receipt.uri
            and state.manifest_object_generation == candidate_receipt.object_generation
            and state.manifest_sha256 == candidate_receipt.sha256
            and state.ops_image_uri == evaluated_ops_image_uri
            and state.source_revision == evaluated_source_revision
        ):
            raise RuntimeError("generation is active with a different immutable receipt")
        print(f"generation {generation_id}@{state.epoch} is already promoted")
        return

    if evaluated_active_generation == "__uninitialized__":
        sentinels_are_exact = (
            evaluated_active_epoch == 0
            and evaluated_active_manifest_generation == 0
            and evaluated_active_manifest_sha256 == "__uninitialized__"
            and evaluated_active_ops_image_uri == "__uninitialized__"
            and evaluated_active_source_revision == "__uninitialized__"
            and evaluated_active_champion_score == -1.0
        )
        if not sentinels_are_exact or state is not None:
            raise RuntimeError("active state appeared after first-generation evaluation")
        initialize_generation(
            bq=bq,
            gcs=gcs,
            binding=binding,
            generation_id=generation_id,
            manifest=manifest,
            item_rows=item_rows,
            receipt=candidate_receipt,
            ops_image_uri=evaluated_ops_image_uri,
            source_revision=evaluated_source_revision,
            score=score,
            owner=owner,
        )
        return

    evaluated_active_generation = validate_generation_id(evaluated_active_generation)
    if evaluated_active_manifest_generation <= 0:
        raise ValueError("evaluated active manifest generation must be positive")
    evaluated_active_receipt_sha256 = active_state_receipt_sha256(
        generation_id=evaluated_active_generation,
        epoch=evaluated_active_epoch,
        manifest_uri_value=manifest_uri(BUCKET, evaluated_active_generation),
        manifest_object_generation=evaluated_active_manifest_generation,
        manifest_sha256=evaluated_active_manifest_sha256,
        ops_image_uri=evaluated_active_ops_image_uri,
        source_revision=evaluated_active_source_revision,
        champion_score=evaluated_active_champion_score,
        binding=binding,
    )
    if (
        state is None
        or state.generation_id != evaluated_active_generation
        or state.epoch != evaluated_active_epoch
        or state.binding != binding
        or state_receipt_sha256(state) != evaluated_active_receipt_sha256
    ):
        raise RuntimeError("active immutable receipt changed after evaluation")
    fence = acquire_lease(
        bq,
        state=state,
        owner=owner,
        candidate_generation=generation_id,
        candidate_receipt=candidate_receipt,
        candidate_ops_image_uri=evaluated_ops_image_uri,
        candidate_source_revision=evaluated_source_revision,
        candidate_score=score,
        evaluated_active_receipt_sha256=evaluated_active_receipt_sha256,
    )
    heartbeat_lease(bq, fence, "binding")
    index, endpoint = matching_resources(binding)
    try:
        verify_live_vertex_target(
            index,
            endpoint,
            expected_index_name=binding.index_name,
            expected_endpoint_name=binding.index_endpoint,
            deployed_index_id=binding.deployed_index_id,
            expected_dimensions=manifest["dimensions"],
            fence=lambda: heartbeat_lease(bq, fence, "binding"),
        )
        heartbeat_lease(bq, fence, "upserting")
        specs = upsert_generation(
            index,
            rows=item_rows,
            generation_id=generation_id,
            fence=lambda: heartbeat_lease(bq, fence, "upserting"),
        )
        heartbeat_lease(bq, fence, "propagating")
        wait_for_generation_visibility(
            endpoint,
            deployed_index_id=binding.deployed_index_id,
            specs=specs,
            generation_id=generation_id,
            canary_datapoint_id=manifest["items"]["canary_datapoint_id"],
            fence=lambda: heartbeat_lease(bq, fence, "propagating"),
        )
        # Re-hash both immutable GCS users and BigQuery staging immediately
        # before the pointer transaction, then fence once more.
        refreshed_manifest, _, _ = load_candidate(
            gcs,
            generation_id,
            binding,
            candidate_receipt,
            evaluated_ops_image_uri,
            evaluated_source_revision,
            fence=lambda: heartbeat_lease(bq, fence, "verifying"),
        )
        if refreshed_manifest != manifest:
            raise RuntimeError("candidate manifest changed during promotion")
        verify_staged_users(
            bq,
            manifest,
            generation_id,
            fence=lambda: heartbeat_lease(bq, fence, "verifying"),
        )
        verify_live_vertex_target(
            index,
            endpoint,
            expected_index_name=binding.index_name,
            expected_endpoint_name=binding.index_endpoint,
            deployed_index_id=binding.deployed_index_id,
            expected_dimensions=manifest["dimensions"],
            fence=lambda: heartbeat_lease(bq, fence, "final-live-binding"),
        )
        heartbeat_lease(bq, fence, "finalizing")
        finalize_promotion(bq, fence)
    except Exception:
        current = read_active_state(bq)
        if (
            current
            and current.generation_id == generation_id
            and current.epoch == fence.lease_epoch
            and current.manifest_uri == candidate_receipt.uri
            and current.manifest_object_generation == candidate_receipt.object_generation
            and current.manifest_sha256 == candidate_receipt.sha256
            and current.ops_image_uri == evaluated_ops_image_uri
            and current.source_revision == evaluated_source_revision
            and current.champion_score == score
            and current.binding == binding
        ):
            print(f"generation {generation_id}@{current.epoch} committed")
            return
        try:
            release_lease(bq, fence)
        except Exception as release_error:
            raise RuntimeError(
                "promotion failed and its lease could not be released; stale recovery required"
            ) from release_error
        raise
    print(
        f"promoted generation {generation_id}@{fence.lease_epoch}; "
        "prior generation remains queryable for in-flight requests and rollback"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--evaluated-active-generation", required=True)
    parser.add_argument("--evaluated-active-epoch", type=int, required=True)
    parser.add_argument("--evaluated-manifest-generation", type=int, required=True)
    parser.add_argument("--evaluated-manifest-sha256", required=True)
    parser.add_argument("--evaluated-ops-image-uri", required=True)
    parser.add_argument("--evaluated-source-revision", required=True)
    parser.add_argument(
        "--evaluated-active-manifest-generation", type=int, required=True
    )
    parser.add_argument("--evaluated-active-manifest-sha256", required=True)
    parser.add_argument("--evaluated-active-ops-image-uri", required=True)
    parser.add_argument("--evaluated-active-source-revision", required=True)
    parser.add_argument(
        "--evaluated-active-champion-score", type=float, required=True
    )
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--index-endpoint", required=True)
    parser.add_argument("--deployed-index-id", required=True)
    args = parser.parse_args()
    promote(
        generation_id=args.generation_id,
        score=args.score,
        evaluated_active_generation=args.evaluated_active_generation,
        evaluated_active_epoch=args.evaluated_active_epoch,
        evaluated_manifest_generation=args.evaluated_manifest_generation,
        evaluated_manifest_sha256=args.evaluated_manifest_sha256,
        evaluated_ops_image_uri=args.evaluated_ops_image_uri,
        evaluated_source_revision=args.evaluated_source_revision,
        evaluated_active_manifest_generation=(
            args.evaluated_active_manifest_generation
        ),
        evaluated_active_manifest_sha256=args.evaluated_active_manifest_sha256,
        evaluated_active_ops_image_uri=args.evaluated_active_ops_image_uri,
        evaluated_active_source_revision=args.evaluated_active_source_revision,
        evaluated_active_champion_score=args.evaluated_active_champion_score,
        binding=validate_binding(
            args.index_name, args.index_endpoint, args.deployed_index_id
        ),
    )


if __name__ == "__main__":
    main()

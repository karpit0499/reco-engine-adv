"""Fail-closed runtime and ANN-fidelity evidence for one active generation.

The default command is a local dry run. Live execution is read/query-only and
requires a plan-specific confirmation phrase. Unit tests inject all cloud
clients; importing this module never obtains credentials or calls Google Cloud.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.generation_contract import (  # noqa: E402
    GENERATION_NAMESPACE,
    SCHEMA_VERSION as PROMOTION_MANIFEST_SCHEMA_VERSION,
    ServingBinding,
    canonical_item_payload,
    namespaced_datapoint_id,
    strip_namespaced_datapoint_id,
    validate_binding,
    validate_generation_id,
    validate_ops_image_uri,
    validate_sha256,
    validate_source_revision,
)


INPUT_SCHEMA_VERSION = 2
OUTPUT_SCHEMA_VERSION = 2
DIAGNOSTIC_SCHEMA_VERSION = 1
INPUT_ARTIFACT_KIND = "production_generation_embedding_snapshot"
MEASUREMENT_KIND = "ann_retrieval_fidelity_and_client_latency"
DIAGNOSTIC_KIND = "vertex_runtime_benchmark_diagnostic"
SOURCE_KIND = "production_interaction_embedding_export"
OPERATOR_ATTESTED = "operator_attested"
MIN_CATALOG_ITEMS = 100
MAX_CATALOG_ITEMS = 100_000
MIN_QUERIES = 20
MAX_QUERIES = 500
MAX_DIMENSIONS = 2_048
MAX_TOP_K = 100
MAX_WARMUPS = 50
MAX_EXACT_SCALAR_OPERATIONS = 500_000_000
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
FIXED_CONTROL_PLANE_CALLS = 11
MAX_POTENTIALLY_BILLED_CALLS = FIXED_CONTROL_PLANE_CALLS + MAX_QUERIES + MAX_WARMUPS
LIVE_CONFIRMATION_TEMPLATE = (
    "EXECUTE {calls} POTENTIALLY BILLED CLOUD API CALLS FOR {generation}@{epoch}"
)
DISTANCE_METRICS = {
    "DOT_PRODUCT_DISTANCE",
    "COSINE_DISTANCE",
    "SQUARED_L2_DISTANCE",
}
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
TABLE_RE = re.compile(
    r"^[a-z][a-z0-9-]{4,28}[a-z0-9]\.reco\.active_embedding_generation$"
)
GS_URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")


class BenchmarkContractError(ValueError):
    """An input, cloud response, or report violated the evidence contract."""


@dataclass(frozen=True)
class VectorRow:
    row_id: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class ManifestIdentity:
    uri: str
    object_generation: int
    sha256: str


@dataclass(frozen=True)
class ExpectedActivePointer:
    table: str
    generation_id: str
    epoch: int
    fence_epoch: int
    previous_generation_id: str | None
    manifest: ManifestIdentity
    ops_image_uri: str
    source_revision: str
    retained_generation_ids: tuple[str, ...]
    champion_score: float
    binding: ServingBinding
    promoted_at: str


@dataclass(frozen=True)
class ManifestContract:
    schema_version: int
    dimensions: int
    item_count: int
    item_sha256: str
    canonical_uri: str
    canonical_object_generation: int


@dataclass(frozen=True)
class RuntimeArtifact:
    expected_active: ExpectedActivePointer
    manifest_contract: ManifestContract
    distance_metric: str
    catalog: tuple[VectorRow, ...]
    queries: tuple[VectorRow, ...]
    created_at: str
    source: Mapping[str, Any]
    catalog_payload_sha256: str


@dataclass(frozen=True)
class BenchmarkConfig:
    binding: ServingBinding
    active_pointer_table: str
    generation_id: str
    promotion_epoch: int
    client_region: str
    top_k: int
    warmups: int
    query_limit: int | None
    artifact_sha256: str
    repository_commit: str | None


@dataclass(frozen=True)
class CloudClients:
    bigquery: Any
    storage: Any
    index_service: Any
    endpoint_service: Any
    query_endpoint: Any
    namespace_factory: Callable[[str, list[str], list[str]], Any]
    sdk_version: str


def _schema_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkContractError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_json(payload: bytes, context: str) -> Any:
    def reject_constant(value: str) -> None:
        raise BenchmarkContractError(f"{context} contains non-finite number {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except BenchmarkContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"{context} is not valid UTF-8 JSON") from exc


def _validate_json_schema(document: Any, schema_name: str, context: str) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise BenchmarkContractError(
            "install benchmarks/vector_search_runtime/requirements.txt first"
        ) from exc
    schema = _decode_json(_schema_path(schema_name).read_bytes(), schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise BenchmarkContractError(f"{context} schema violation at {location}")


def _exact_object(value: Any, required: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{context} must be a JSON object")
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise BenchmarkContractError(
            f"{context} keys differ from the contract; missing={missing}, extra={extra}"
        )
    return value


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkContractError(f"{context} must be a positive integer")
    return value


def _timestamp(value: Any, context: str, *, require_z: bool = True) -> str:
    if not isinstance(value, str) or (require_z and not value.endswith("Z")):
        raise BenchmarkContractError(f"{context} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BenchmarkContractError(
            f"{context} must be an RFC 3339 UTC timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise BenchmarkContractError(f"{context} must use UTC")
    return value


def _format_timestamp(value: Any, context: str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return _timestamp(value, context)


def _vector_rows(
    rows: Any,
    *,
    dimensions: int,
    minimum: int,
    maximum: int,
    context: str,
) -> tuple[VectorRow, ...]:
    if not isinstance(rows, list) or not minimum <= len(rows) <= maximum:
        raise BenchmarkContractError(
            f"{context} must contain between {minimum} and {maximum} rows"
        )
    normalized: list[VectorRow] = []
    seen: set[str] = set()
    for position, raw in enumerate(rows):
        row = _exact_object(raw, {"id", "embedding"}, f"{context}[{position}]")
        row_id = row["id"]
        if not isinstance(row_id, str) or not SAFE_LABEL.fullmatch(row_id):
            raise BenchmarkContractError(f"{context}[{position}].id is invalid")
        if row_id in seen:
            raise BenchmarkContractError(f"{context} contains a duplicate id")
        seen.add(row_id)
        embedding = row["embedding"]
        if not isinstance(embedding, list) or len(embedding) != dimensions:
            raise BenchmarkContractError(
                f"{context}[{position}].embedding must have dimension {dimensions}"
            )
        if any(isinstance(component, bool) for component in embedding):
            raise BenchmarkContractError(
                f"{context}[{position}].embedding must be numeric"
            )
        try:
            vector = tuple(float(component) for component in embedding)
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError(
                f"{context}[{position}].embedding must be numeric"
            ) from exc
        if not all(math.isfinite(component) for component in vector):
            raise BenchmarkContractError(
                f"{context}[{position}].embedding must contain finite values"
            )
        normalized.append(VectorRow(row_id, vector))
    return tuple(normalized)


def _parse_manifest_identity(value: Any, context: str) -> ManifestIdentity:
    value = _exact_object(value, {"uri", "object_generation", "sha256"}, context)
    uri = value["uri"]
    if not isinstance(uri, str) or not GS_URI_RE.fullmatch(uri):
        raise BenchmarkContractError(f"{context}.uri must be a gs:// object URI")
    try:
        digest = validate_sha256(value["sha256"], f"{context}.sha256")
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(str(exc)) from exc
    return ManifestIdentity(
        uri=uri,
        object_generation=_positive_int(
            value["object_generation"], f"{context}.object_generation"
        ),
        sha256=digest,
    )


def _pointer_as_dict(pointer: ExpectedActivePointer) -> dict[str, Any]:
    return {
        "table": pointer.table,
        "generation_id": pointer.generation_id,
        "epoch": pointer.epoch,
        "fence_epoch": pointer.fence_epoch,
        "previous_generation_id": pointer.previous_generation_id,
        "manifest": {
            "uri": pointer.manifest.uri,
            "object_generation": pointer.manifest.object_generation,
            "sha256": pointer.manifest.sha256,
        },
        "provenance": {
            "ops_image_uri": pointer.ops_image_uri,
            "source_revision": pointer.source_revision,
        },
        "retained_generation_ids": list(pointer.retained_generation_ids),
        "champion_score": pointer.champion_score,
        "serving": {
            "index_resource": pointer.binding.index_name,
            "index_endpoint": pointer.binding.index_endpoint,
            "deployed_index_id": pointer.binding.deployed_index_id,
        },
        "promoted_at": pointer.promoted_at,
    }


def _parse_active_pointer(value: Any, context: str) -> ExpectedActivePointer:
    value = _exact_object(
        value,
        {
            "table",
            "generation_id",
            "epoch",
            "fence_epoch",
            "previous_generation_id",
            "manifest",
            "provenance",
            "retained_generation_ids",
            "champion_score",
            "serving",
            "promoted_at",
        },
        context,
    )
    table = value["table"]
    if not isinstance(table, str) or not TABLE_RE.fullmatch(table):
        raise BenchmarkContractError(f"{context}.table is not the promotion pointer table")
    try:
        generation_id = validate_generation_id(value["generation_id"])
        previous = value["previous_generation_id"]
        if previous is not None:
            previous = validate_generation_id(previous)
        serving = _exact_object(
            value["serving"],
            {"index_resource", "index_endpoint", "deployed_index_id"},
            f"{context}.serving",
        )
        binding = validate_binding(
            serving["index_resource"],
            serving["index_endpoint"],
            serving["deployed_index_id"],
        )
        provenance = _exact_object(
            value["provenance"],
            {"ops_image_uri", "source_revision"},
            f"{context}.provenance",
        )
        ops_image_uri = validate_ops_image_uri(provenance["ops_image_uri"])
        source_revision = validate_source_revision(provenance["source_revision"])
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(str(exc)) from exc
    project = binding.index_name.split("/")[1]
    if table != f"{project}.reco.active_embedding_generation":
        raise BenchmarkContractError(
            f"{context}.table must belong to the exact Vertex project"
        )
    epoch = _positive_int(value["epoch"], f"{context}.epoch")
    fence_epoch = _positive_int(value["fence_epoch"], f"{context}.fence_epoch")
    if fence_epoch < epoch:
        raise BenchmarkContractError(f"{context}.fence_epoch moved backwards")
    retained = value["retained_generation_ids"]
    if not isinstance(retained, list) or not retained:
        raise BenchmarkContractError(f"{context}.retained_generation_ids cannot be empty")
    try:
        retained_ids = tuple(validate_generation_id(item) for item in retained)
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(str(exc)) from exc
    if len(set(retained_ids)) != len(retained_ids) or generation_id not in retained_ids:
        raise BenchmarkContractError(
            f"{context}.retained_generation_ids must be unique and contain the active generation"
        )
    champion = value["champion_score"]
    if (
        isinstance(champion, bool)
        or not isinstance(champion, (int, float))
        or not math.isfinite(float(champion))
        or not 0.0 <= float(champion) <= 1.0
    ):
        raise BenchmarkContractError(f"{context}.champion_score is invalid")
    manifest = _parse_manifest_identity(value["manifest"], f"{context}.manifest")
    expected_suffix = f"/candidate-generations/{generation_id}/manifest.json"
    if not manifest.uri.endswith(expected_suffix):
        raise BenchmarkContractError(
            f"{context}.manifest URI is not generation-specific"
        )
    return ExpectedActivePointer(
        table=table,
        generation_id=generation_id,
        epoch=epoch,
        fence_epoch=fence_epoch,
        previous_generation_id=previous,
        manifest=manifest,
        ops_image_uri=ops_image_uri,
        source_revision=source_revision,
        retained_generation_ids=retained_ids,
        champion_score=float(champion),
        binding=binding,
        promoted_at=_timestamp(value["promoted_at"], f"{context}.promoted_at"),
    )


def parse_artifact(document: Any) -> RuntimeArtifact:
    """Validate and hash-bind a private production-generation snapshot."""

    _validate_json_schema(document, "input.schema.json", "artifact")
    root = _exact_object(
        document,
        {
            "schema_version",
            "artifact_kind",
            "created_at",
            "source",
            "expected_active_pointer",
            "manifest_contract",
            "catalog",
            "queries",
        },
        "artifact",
    )
    if root["schema_version"] != INPUT_SCHEMA_VERSION:
        raise BenchmarkContractError("unsupported input schema_version")
    if root["artifact_kind"] != INPUT_ARTIFACT_KIND:
        raise BenchmarkContractError("artifact_kind is not a production snapshot")
    created_at = _timestamp(root["created_at"], "artifact.created_at")
    source = _exact_object(
        root["source"],
        {
            "kind",
            "contains_synthetic_data",
            "description",
            "interaction_count",
            "provenance_verification",
        },
        "artifact.source",
    )
    if source["kind"] != SOURCE_KIND or source["contains_synthetic_data"] is not False:
        raise BenchmarkContractError(
            "runtime evidence requires an explicitly non-synthetic production export"
        )
    if source["provenance_verification"] != OPERATOR_ATTESTED:
        raise BenchmarkContractError(
            "source provenance must be labeled operator_attested"
        )
    _positive_int(source["interaction_count"], "artifact.source.interaction_count")

    expected_active = _parse_active_pointer(
        root["expected_active_pointer"], "artifact.expected_active_pointer"
    )
    contract = _exact_object(
        root["manifest_contract"],
        {"schema_version", "dimensions", "items"},
        "artifact.manifest_contract",
    )
    if contract["schema_version"] != PROMOTION_MANIFEST_SCHEMA_VERSION:
        raise BenchmarkContractError("manifest contract schema does not match promotion code")
    dimensions = _positive_int(
        contract["dimensions"], "artifact.manifest_contract.dimensions"
    )
    if dimensions > MAX_DIMENSIONS:
        raise BenchmarkContractError("manifest dimensions exceed the hard cap")
    manifest_items = _exact_object(
        contract["items"],
        {"count", "sha256", "canonical_uri", "canonical_object_generation"},
        "artifact.manifest_contract.items",
    )
    item_count = _positive_int(
        manifest_items["count"], "artifact.manifest_contract.items.count"
    )
    if not MIN_CATALOG_ITEMS <= item_count <= MAX_CATALOG_ITEMS:
        raise BenchmarkContractError("manifest item count is outside the hard caps")
    try:
        item_sha256 = validate_sha256(
            manifest_items["sha256"], "artifact.manifest_contract.items.sha256"
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(str(exc)) from exc
    canonical_uri = manifest_items["canonical_uri"]
    if not isinstance(canonical_uri, str) or not GS_URI_RE.fullmatch(canonical_uri):
        raise BenchmarkContractError("manifest canonical item URI is invalid")
    manifest_contract = ManifestContract(
        schema_version=contract["schema_version"],
        dimensions=dimensions,
        item_count=item_count,
        item_sha256=item_sha256,
        canonical_uri=canonical_uri,
        canonical_object_generation=_positive_int(
            manifest_items["canonical_object_generation"],
            "artifact.manifest_contract.items.canonical_object_generation",
        ),
    )

    catalog = _exact_object(root["catalog"], {"distance_metric", "items"}, "artifact.catalog")
    distance_metric = catalog["distance_metric"]
    if distance_metric not in DISTANCE_METRICS:
        raise BenchmarkContractError("artifact.catalog.distance_metric is unsupported")
    items = _vector_rows(
        catalog["items"],
        dimensions=dimensions,
        minimum=MIN_CATALOG_ITEMS,
        maximum=MAX_CATALOG_ITEMS,
        context="artifact.catalog.items",
    )
    if len(items) != item_count:
        raise BenchmarkContractError(
            "catalog is incomplete: row count differs from the promotion manifest"
        )
    canonical_payload, _ = canonical_item_payload(
        ({"id": row.row_id, "embedding": list(row.embedding)} for row in items),
        dimensions=dimensions,
    )
    catalog_digest = hashlib.sha256(canonical_payload).hexdigest()
    if catalog_digest != item_sha256:
        raise BenchmarkContractError(
            "catalog is incomplete or changed: canonical SHA-256 differs from the manifest"
        )
    queries = _vector_rows(
        root["queries"],
        dimensions=dimensions,
        minimum=MIN_QUERIES,
        maximum=MAX_QUERIES,
        context="artifact.queries",
    )
    if source["interaction_count"] < max(len(items), len(queries)):
        raise BenchmarkContractError(
            "interaction_count cannot be smaller than the catalog or query set"
        )
    if distance_metric == "COSINE_DISTANCE":
        for row in (*items, *queries):
            if math.isclose(sum(value * value for value in row.embedding), 0.0):
                raise BenchmarkContractError(
                    "cosine-distance vectors must have non-zero norm"
                )

    return RuntimeArtifact(
        expected_active=expected_active,
        manifest_contract=manifest_contract,
        distance_metric=distance_metric,
        catalog=items,
        queries=queries,
        created_at=created_at,
        source=dict(source),
        catalog_payload_sha256=catalog_digest,
    )


def load_artifact(path: Path) -> tuple[RuntimeArtifact, str]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise BenchmarkContractError("input artifact could not be read") from exc
    if size <= 0 or size > MAX_ARTIFACT_BYTES:
        raise BenchmarkContractError("input artifact size is outside the hard caps")
    raw = path.read_bytes()
    return parse_artifact(_decode_json(raw, "input artifact")), hashlib.sha256(raw).hexdigest()


def validate_config(config: BenchmarkConfig, artifact: RuntimeArtifact) -> None:
    if config.binding != artifact.expected_active.binding:
        raise BenchmarkContractError("command-line Vertex binding differs from the pointer")
    if config.active_pointer_table != artifact.expected_active.table:
        raise BenchmarkContractError("command-line pointer table differs from the artifact")
    if config.generation_id != artifact.expected_active.generation_id:
        raise BenchmarkContractError("command-line generation differs from the pointer")
    if config.promotion_epoch != artifact.expected_active.epoch:
        raise BenchmarkContractError("command-line epoch differs from the pointer")
    if not isinstance(config.client_region, str) or not SAFE_LABEL.fullmatch(config.client_region):
        raise BenchmarkContractError("client_region is invalid")
    if not 1 <= config.top_k <= min(MAX_TOP_K, len(artifact.catalog)):
        raise BenchmarkContractError("top_k is outside the hard caps")
    if not 0 <= config.warmups <= MAX_WARMUPS:
        raise BenchmarkContractError("warmups are outside the hard caps")
    if config.query_limit is not None and not MIN_QUERIES <= config.query_limit <= MAX_QUERIES:
        raise BenchmarkContractError("query_limit is outside the hard caps")
    selected_queries = artifact.queries[: config.query_limit]
    if not MIN_QUERIES <= len(selected_queries) <= MAX_QUERIES:
        raise BenchmarkContractError("measured query count is outside the hard caps")
    exact_operations = (
        len(selected_queries) * len(artifact.catalog) * artifact.manifest_contract.dimensions
    )
    if exact_operations > MAX_EXACT_SCALAR_OPERATIONS:
        raise BenchmarkContractError("exact-scan workload exceeds the hard compute cap")
    planned_calls = FIXED_CONTROL_PLANE_CALLS + config.warmups + len(selected_queries)
    if planned_calls > MAX_POTENTIALLY_BILLED_CALLS:
        raise BenchmarkContractError("planned cloud calls exceed the hard billing cap")
    if config.repository_commit is not None and not re.fullmatch(
        r"[0-9a-f]{40}", config.repository_commit
    ):
        raise BenchmarkContractError("repository_commit must be an exact commit")


def build_plan(config: BenchmarkConfig, artifact: RuntimeArtifact) -> dict[str, Any]:
    queries = len(artifact.queries[: config.query_limit])
    find_calls = config.warmups + queries
    total_calls = FIXED_CONTROL_PLANE_CALLS + find_calls
    confirmation = LIVE_CONFIRMATION_TEMPLATE.format(
        calls=total_calls,
        generation=config.generation_id,
        epoch=config.promotion_epoch,
    )
    return {
        "dry_run_default": True,
        "planned_cloud_api_calls": {
            "bigquery_active_pointer_queries": 2,
            "gcs_manifest_metadata_reads": 2,
            "gcs_generation_pinned_manifest_downloads": 2,
            "vertex_index_metadata_reads": 2,
            "vertex_endpoint_metadata_reads": 3,
            "vertex_find_neighbors_calls": find_calls,
            "total_potentially_billable_calls": total_calls,
        },
        "planned_exact_scalar_operations": (
            queries * len(artifact.catalog) * artifact.manifest_contract.dimensions
        ),
        "hard_caps": {
            "catalog_items": MAX_CATALOG_ITEMS,
            "measured_queries": MAX_QUERIES,
            "warmups": MAX_WARMUPS,
            "top_k": MAX_TOP_K,
            "exact_scalar_operations": MAX_EXACT_SCALAR_OPERATIONS,
            "potentially_billable_calls": MAX_POTENTIALLY_BILLED_CALLS,
        },
        "required_confirmation_phrase": confirmation,
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _row_value(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError):
        return getattr(row, name, None)


def read_active_pointer(bigquery_client: Any, expected: ExpectedActivePointer) -> ExpectedActivePointer:
    """Read the stable active pointer once and reject an in-progress promotion."""

    if not TABLE_RE.fullmatch(expected.table):
        raise BenchmarkContractError("active pointer table is invalid")
    query = f"""SELECT generation_id, epoch, previous_generation_id,
        manifest_uri, manifest_object_generation, manifest_sha256,
        ops_image_uri, source_revision, retained_generation_ids, champion_score,
        index_name, index_endpoint, deployed_index_id, promoted_at, lease_epoch,
        lease_owner, lease_phase, pending_generation_id, pending_manifest_uri,
        pending_manifest_object_generation, pending_manifest_sha256,
        pending_ops_image_uri, pending_source_revision, pending_score
      FROM `{expected.table}` WHERE singleton = TRUE"""
    try:
        rows = list(bigquery_client.query(query).result())
    except Exception as exc:
        raise BenchmarkContractError("active BigQuery pointer could not be read") from exc
    if len(rows) != 1:
        raise BenchmarkContractError("active pointer must contain exactly one singleton row")
    row = rows[0]
    transient_fields = (
        "lease_owner",
        "lease_phase",
        "pending_generation_id",
        "pending_manifest_uri",
        "pending_manifest_object_generation",
        "pending_manifest_sha256",
        "pending_ops_image_uri",
        "pending_source_revision",
        "pending_score",
    )
    if any(_row_value(row, field) is not None for field in transient_fields):
        raise BenchmarkContractError("active pointer has an in-progress promotion lease")
    retained = _row_value(row, "retained_generation_ids")
    document = {
        "table": expected.table,
        "generation_id": str(_row_value(row, "generation_id")),
        "epoch": int(_row_value(row, "epoch")),
        "fence_epoch": int(_row_value(row, "lease_epoch")),
        "previous_generation_id": _row_value(row, "previous_generation_id"),
        "manifest": {
            "uri": str(_row_value(row, "manifest_uri")),
            "object_generation": int(_row_value(row, "manifest_object_generation")),
            "sha256": str(_row_value(row, "manifest_sha256")),
        },
        "provenance": {
            "ops_image_uri": str(_row_value(row, "ops_image_uri")),
            "source_revision": str(_row_value(row, "source_revision")),
        },
        "retained_generation_ids": list(retained or []),
        "champion_score": _row_value(row, "champion_score"),
        "serving": {
            "index_resource": str(_row_value(row, "index_name")),
            "index_endpoint": str(_row_value(row, "index_endpoint")),
            "deployed_index_id": str(_row_value(row, "deployed_index_id")),
        },
        "promoted_at": _format_timestamp(_row_value(row, "promoted_at"), "promoted_at"),
    }
    actual = _parse_active_pointer(document, "live active pointer")
    if actual != expected:
        raise BenchmarkContractError("live active pointer differs from the private input")
    return actual


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    match = GS_URI_RE.fullmatch(uri)
    if not match or not match.group(2) or match.group(2).endswith("/"):
        raise BenchmarkContractError("manifest identity has an invalid GCS object URI")
    return match.group(1), match.group(2)


def read_immutable_manifest(storage_client: Any, identity: ManifestIdentity) -> bytes:
    bucket_name, object_name = _parse_gs_uri(identity.uri)
    try:
        blob = storage_client.bucket(bucket_name).blob(object_name)
        blob.reload()
        observed_generation = int(blob.generation)
        if observed_generation != identity.object_generation:
            raise BenchmarkContractError("manifest object generation differs from the pointer")
        payload = blob.download_as_bytes(if_generation_match=identity.object_generation)
    except BenchmarkContractError:
        raise
    except Exception as exc:
        raise BenchmarkContractError("immutable promotion manifest could not be read") from exc
    if hashlib.sha256(payload).hexdigest() != identity.sha256:
        raise BenchmarkContractError("promotion manifest digest differs from the pointer")
    return payload


def parse_live_promotion_manifest(
    payload: bytes,
    *,
    artifact: RuntimeArtifact,
) -> dict[str, Any]:
    """Strictly validate the promotion v3 manifest and catalog commitments."""

    root = _exact_object(
        _decode_json(payload, "promotion manifest"),
        {
            "schema_version",
            "generation_id",
            "created_at",
            "trained_at",
            "dimensions",
            "serving",
            "provenance",
            "items",
            "users",
            "models",
        },
        "promotion manifest",
    )
    pointer = artifact.expected_active
    contract = artifact.manifest_contract
    if root["schema_version"] != contract.schema_version:
        raise BenchmarkContractError("live manifest schema version differs from the input")
    if root["generation_id"] != pointer.generation_id:
        raise BenchmarkContractError("live manifest generation differs from the pointer")
    _timestamp(root["created_at"], "promotion manifest.created_at", require_z=False)
    _timestamp(root["trained_at"], "promotion manifest.trained_at", require_z=False)
    if root["dimensions"] != contract.dimensions:
        raise BenchmarkContractError("live manifest dimensions differ from the input")
    serving = _exact_object(
        root["serving"],
        {"index_name", "index_endpoint", "deployed_index_id", "generation_namespace"},
        "promotion manifest.serving",
    )
    if serving != {
        "index_name": pointer.binding.index_name,
        "index_endpoint": pointer.binding.index_endpoint,
        "deployed_index_id": pointer.binding.deployed_index_id,
        "generation_namespace": GENERATION_NAMESPACE,
    }:
        raise BenchmarkContractError("live manifest serving binding differs from the pointer")
    provenance = _exact_object(
        root["provenance"],
        {"ops_image_uri", "source_revision"},
        "promotion manifest.provenance",
    )
    if provenance != {
        "ops_image_uri": pointer.ops_image_uri,
        "source_revision": pointer.source_revision,
    }:
        raise BenchmarkContractError("live manifest provenance differs from the pointer")
    items = _exact_object(
        root["items"],
        {
            "count",
            "sha256",
            "canonical_uri",
            "canonical_object_generation",
            "regional_uri",
            "regional_object_generation",
            "canary_item_id",
            "canary_datapoint_id",
        },
        "promotion manifest.items",
    )
    expected_item_values = {
        "count": contract.item_count,
        "sha256": contract.item_sha256,
        "canonical_uri": contract.canonical_uri,
        "canonical_object_generation": contract.canonical_object_generation,
    }
    if any(items[key] != value for key, value in expected_item_values.items()):
        raise BenchmarkContractError("live manifest item commitment differs from the input")
    _positive_int(items["regional_object_generation"], "manifest regional generation")
    if not isinstance(items["regional_uri"], str) or not GS_URI_RE.fullmatch(items["regional_uri"]):
        raise BenchmarkContractError("live manifest regional item URI is invalid")
    if not isinstance(items["canary_item_id"], str) or not items["canary_item_id"]:
        raise BenchmarkContractError("live manifest item canary is invalid")
    if items["canary_datapoint_id"] != namespaced_datapoint_id(
        pointer.generation_id, items["canary_item_id"]
    ):
        raise BenchmarkContractError("live manifest canary datapoint is invalid")
    users = _exact_object(
        root["users"],
        {
            "count",
            "dimensions",
            "sha256",
            "uri",
            "object_generation",
            "table",
            "generation_id",
            "canary_user_id",
        },
        "promotion manifest.users",
    )
    _positive_int(users["count"], "promotion manifest.users.count")
    _positive_int(users["object_generation"], "promotion manifest.users.object_generation")
    try:
        validate_sha256(users["sha256"], "promotion manifest.users.sha256")
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(str(exc)) from exc
    if (
        users["dimensions"] != contract.dimensions
        or users["generation_id"] != pointer.generation_id
        or not isinstance(users["uri"], str)
        or not GS_URI_RE.fullmatch(users["uri"])
        or not isinstance(users["table"], str)
        or not users["table"].endswith(".reco.user_embedding_generations")
        or not isinstance(users["canary_user_id"], str)
        or not users["canary_user_id"]
    ):
        raise BenchmarkContractError("live manifest user contract is invalid")
    models = _exact_object(
        root["models"],
        {"user_tower_uri", "item_tower_uri"},
        "promotion manifest.models",
    )
    if any(not isinstance(value, str) or not value for value in models.values()):
        raise BenchmarkContractError("live manifest model URIs are invalid")
    return {
        "identity": {
            "uri": pointer.manifest.uri,
            "object_generation": pointer.manifest.object_generation,
            "sha256": pointer.manifest.sha256,
        },
        "schema_version": root["schema_version"],
        "generation_id": pointer.generation_id,
        "dimensions": contract.dimensions,
        "items": {
            "count": contract.item_count,
            "sha256": contract.item_sha256,
            "canonical_uri": contract.canonical_uri,
            "canonical_object_generation": contract.canonical_object_generation,
            "catalog_payload_match": True,
        },
        "provenance": dict(provenance),
    }


def _protobuf_mapping(message: Any, context: str) -> Mapping[str, Any]:
    if isinstance(message, Mapping):
        return message
    try:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(message, preserving_proto_field_name=False)
    except Exception as exc:
        raise BenchmarkContractError(f"{context} metadata could not be decoded") from exc


def read_live_vertex_contract(
    index_service: Any,
    endpoint_service: Any,
    *,
    binding: ServingBinding,
    dimensions: int,
    distance_metric: str,
) -> dict[str, Any]:
    """Read and validate the live Index and deployed-index mapping once."""

    try:
        index_doc = _protobuf_mapping(
            index_service.get_index(name=binding.index_name), "Vertex index"
        )
        endpoint_doc = _protobuf_mapping(
            endpoint_service.get_index_endpoint(name=binding.index_endpoint),
            "Vertex index endpoint",
        )
    except BenchmarkContractError:
        raise
    except Exception as exc:
        raise BenchmarkContractError("live Vertex metadata could not be read") from exc
    if index_doc.get("name") != binding.index_name:
        raise BenchmarkContractError("Vertex index client returned a different resource")
    metadata = index_doc.get("metadata")
    if not isinstance(metadata, Mapping):
        raise BenchmarkContractError("Vertex index metadata is absent")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise BenchmarkContractError("Vertex index config is absent")
    live_dimensions = config.get("dimensions")
    if isinstance(live_dimensions, str) and live_dimensions.isdigit():
        live_dimensions = int(live_dimensions)
    live_distance = config.get("distanceMeasureType") or config.get("distance_measure_type")
    update_method = index_doc.get("indexUpdateMethod") or index_doc.get("index_update_method")
    if live_dimensions != dimensions:
        raise BenchmarkContractError("live Vertex dimension differs from the manifest")
    if live_distance != distance_metric:
        raise BenchmarkContractError("live Vertex distance metric differs from the input")
    if update_method != "STREAM_UPDATE":
        raise BenchmarkContractError("live Vertex index is not configured for STREAM_UPDATE")
    endpoint_name = endpoint_doc.get("name")
    if endpoint_name != binding.index_endpoint:
        raise BenchmarkContractError("Vertex endpoint client returned a different resource")
    deployed = endpoint_doc.get("deployedIndexes") or endpoint_doc.get("deployed_indexes")
    if not isinstance(deployed, list):
        raise BenchmarkContractError("Vertex deployed-index metadata is absent")
    matches = []
    for entry in deployed:
        if not isinstance(entry, Mapping):
            continue
        deployed_id = entry.get("id")
        index_name = entry.get("index")
        if deployed_id == binding.deployed_index_id:
            matches.append(index_name)
    if matches != [binding.index_name]:
        raise BenchmarkContractError(
            "deployed_index_id is not mapped exactly once to the required index"
        )
    return {
        "index_resource": binding.index_name,
        "index_endpoint": binding.index_endpoint,
        "deployed_index_id": binding.deployed_index_id,
        "dimensions": dimensions,
        "distance_metric": distance_metric,
        "index_update_method": "STREAM_UPDATE",
        "deployed_mapping_verified": True,
        "endpoint_region": binding.index_endpoint.split("/")[3],
    }


def capture_control_snapshot(
    cloud: CloudClients,
    artifact: RuntimeArtifact,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now = now or (lambda: datetime.now(timezone.utc))
    pointer = read_active_pointer(cloud.bigquery, artifact.expected_active)
    manifest_payload = read_immutable_manifest(cloud.storage, pointer.manifest)
    manifest = parse_live_promotion_manifest(manifest_payload, artifact=artifact)
    vertex = read_live_vertex_contract(
        cloud.index_service,
        cloud.endpoint_service,
        binding=pointer.binding,
        dimensions=artifact.manifest_contract.dimensions,
        distance_metric=artifact.distance_metric,
    )
    pointer_document = _pointer_as_dict(pointer)
    return {
        "observed_at": now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "active_pointer": pointer_document,
        "active_pointer_sha256": _canonical_sha256(pointer_document),
        "promotion_manifest": manifest,
        "vertex": vertex,
    }


def _control_identity(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "active_pointer": snapshot["active_pointer"],
        "active_pointer_sha256": snapshot["active_pointer_sha256"],
        "promotion_manifest": snapshot["promotion_manifest"],
        "vertex": snapshot["vertex"],
    }


def verify_query_endpoint_binding(endpoint: Any, binding: ServingBinding) -> None:
    if getattr(endpoint, "resource_name", None) != binding.index_endpoint:
        raise BenchmarkContractError("query client is bound to an unexpected endpoint")


def _score(query: Sequence[float], item: Sequence[float], distance_metric: str) -> float:
    if distance_metric == "DOT_PRODUCT_DISTANCE":
        return sum(left * right for left, right in zip(query, item))
    if distance_metric == "COSINE_DISTANCE":
        dot = sum(left * right for left, right in zip(query, item))
        q_norm = math.sqrt(sum(value * value for value in query))
        i_norm = math.sqrt(sum(value * value for value in item))
        return dot / (q_norm * i_norm)
    return sum((left - right) ** 2 for left, right in zip(query, item))


def exact_top_k(
    query: Sequence[float],
    catalog: Sequence[VectorRow],
    distance_metric: str,
    top_k: int,
) -> list[str]:
    scored = [
        (_score(query, item.embedding, distance_metric), item.row_id)
        for item in catalog
    ]
    if distance_metric == "SQUARED_L2_DISTANCE":
        scored.sort(key=lambda pair: (pair[0], pair[1]))
    else:
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [item_id for _, item_id in scored[:top_k]]


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise BenchmarkContractError("cannot compute a percentile without successes")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def environment_metadata(sdk_version: str) -> dict[str, Any]:
    uname = platform.uname()
    return {
        "os": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
        },
        "cpu": {
            "processor": uname.processor or platform.processor() or "unknown",
            "machine": uname.machine,
            "logical_count": os.cpu_count(),
        },
        "python": platform.python_version(),
        "google_cloud_aiplatform_sdk": sdk_version,
    }


def _error_record(exc: Exception, stage: str) -> dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__}


def _query_once(
    *,
    endpoint: Any,
    namespace_factory: Callable[[str, list[str], list[str]], Any],
    binding: ServingBinding,
    generation_id: str,
    query: VectorRow,
    exact_ids: set[str],
    catalog_ids: set[str],
    top_k: int,
    phase: str,
    sequence: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    try:
        namespace = namespace_factory(GENERATION_NAMESPACE, [generation_id], [])
        response = endpoint.find_neighbors(
            deployed_index_id=binding.deployed_index_id,
            queries=[list(query.embedding)],
            num_neighbors=top_k,
            filter=[namespace],
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not isinstance(response, (list, tuple)) or len(response) != 1:
            raise BenchmarkContractError("Vertex response must contain exactly one query")
        neighbors = response[0]
        if not isinstance(neighbors, (list, tuple)) or len(neighbors) != top_k:
            raise BenchmarkContractError("Vertex response did not return exactly top_k rows")
        ids = [
            strip_namespaced_datapoint_id(generation_id, getattr(match, "id", None))
            for match in neighbors
        ]
        if len(set(ids)) != top_k:
            raise BenchmarkContractError("Vertex response contains duplicate datapoint IDs")
        if any(item_id not in catalog_ids for item_id in ids):
            raise BenchmarkContractError("Vertex returned an item outside the complete catalog")
        overlap_count = len(exact_ids.intersection(ids))
        return {
            "phase": phase,
            "sequence": sequence,
            "success": True,
            "latency_ms": round(elapsed_ms, 6),
            "exact_ann_overlap_count": overlap_count,
            "exact_ann_overlap_at_k": overlap_count / top_k,
            "error": None,
        }
    except Exception as exc:  # Failure is retained as evidence; no exact fallback.
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "phase": phase,
            "sequence": sequence,
            "success": False,
            "latency_ms": round(elapsed_ms, 6),
            "exact_ann_overlap_count": None,
            "exact_ann_overlap_at_k": None,
            "error": _error_record(exc, "querying"),
        }


def _manifest_contract_as_dict(contract: ManifestContract) -> dict[str, Any]:
    return {
        "schema_version": contract.schema_version,
        "dimensions": contract.dimensions,
        "items": {
            "count": contract.item_count,
            "sha256": contract.item_sha256,
            "canonical_uri": contract.canonical_uri,
            "canonical_object_generation": contract.canonical_object_generation,
        },
    }


def run_benchmark(
    *,
    config: BenchmarkConfig,
    artifact: RuntimeArtifact,
    endpoint: Any,
    namespace_factory: Callable[[str, list[str], list[str]], Any],
    sdk_version: str,
    control_reader: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Measure one generation and re-fence all control state after queries."""

    validate_config(config, artifact)
    if config.repository_commit is None:
        raise BenchmarkContractError("live evidence requires an exact repository commit")
    verify_query_endpoint_binding(endpoint, config.binding)
    before = control_reader()
    queries = artifact.queries[: config.query_limit]
    exact = {
        query.row_id: set(
            exact_top_k(query.embedding, artifact.catalog, artifact.distance_metric, config.top_k)
        )
        for query in queries
    }
    catalog_ids = {item.row_id for item in artifact.catalog}
    samples: list[dict[str, Any]] = []
    for sequence in range(config.warmups):
        query = queries[sequence % len(queries)]
        samples.append(
            _query_once(
                endpoint=endpoint,
                namespace_factory=namespace_factory,
                binding=config.binding,
                generation_id=config.generation_id,
                query=query,
                exact_ids=exact[query.row_id],
                catalog_ids=catalog_ids,
                top_k=config.top_k,
                phase="warmup",
                sequence=sequence,
            )
        )
    for sequence, query in enumerate(queries):
        samples.append(
            _query_once(
                endpoint=endpoint,
                namespace_factory=namespace_factory,
                binding=config.binding,
                generation_id=config.generation_id,
                query=query,
                exact_ids=exact[query.row_id],
                catalog_ids=catalog_ids,
                top_k=config.top_k,
                phase="measured",
                sequence=sequence,
            )
        )

    postflight_error: Exception | None = None
    after: dict[str, Any] | None = None
    unchanged: bool | None = None
    try:
        after = control_reader()
        unchanged = _control_identity(before) == _control_identity(after)
        if not unchanged:
            raise BenchmarkContractError("active pointer, manifest, or Vertex target changed")
    except Exception as exc:
        postflight_error = exc
        unchanged = False if after is not None else None

    warmup_samples = [sample for sample in samples if sample["phase"] == "warmup"]
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    measured_successes = [sample for sample in measured if sample["success"]]
    warmup_failures = sum(not sample["success"] for sample in warmup_samples)
    measured_failures = sum(not sample["success"] for sample in measured)
    successful_latencies = [sample["latency_ms"] for sample in measured_successes]
    overlaps = [sample["exact_ann_overlap_at_k"] for sample in measured_successes]
    passed = warmup_failures == 0 and measured_failures == 0 and postflight_error is None
    failure: dict[str, str] | None = None
    if postflight_error is not None:
        failure = _error_record(postflight_error, "control_postflight")
    elif warmup_failures or measured_failures:
        failure = {"stage": "querying", "type": "QueryCallFailed"}

    report = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "status": "passed" if passed else "failed",
        "failure": failure,
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "recommendation_quality_evaluated": False,
        "input": {
            "artifact_sha256": config.artifact_sha256,
            "artifact_created_at": artifact.created_at,
            "source": dict(artifact.source),
            "repository_commit": config.repository_commit,
            "expected_active_pointer": _pointer_as_dict(artifact.expected_active),
            "manifest_contract": _manifest_contract_as_dict(artifact.manifest_contract),
        },
        "operator_attestations": {
            "source_provenance": {
                "verification": OPERATOR_ATTESTED,
                "value": artifact.source["description"],
            },
            "client_region": {
                "verification": OPERATOR_ATTESTED,
                "value": config.client_region,
            },
        },
        "plan": build_plan(config, artifact),
        "control_plane": {
            "before": before,
            "after": after,
            "unchanged": unchanged,
        },
        "workload": {
            "catalog_size": len(artifact.catalog),
            "catalog_canonical_sha256": artifact.catalog_payload_sha256,
            "catalog_manifest_match": True,
            "dimensions": artifact.manifest_contract.dimensions,
            "distance_metric": artifact.distance_metric,
            "top_k": config.top_k,
            "warmups_requested": config.warmups,
            "warmups_success": len(warmup_samples) - warmup_failures,
            "warmups_failure": warmup_failures,
            "queries_requested": len(measured),
            "queries_success": len(measured_successes),
            "queries_failure": measured_failures,
        },
        "client_environment": environment_metadata(sdk_version),
        "latency_ms": {
            "scope": "client find_neighbors call wall time; exact scoring excluded",
            "successful_queries_only": True,
            "p50": round(_percentile(successful_latencies, 0.50), 6)
            if successful_latencies
            else None,
            "p95": round(_percentile(successful_latencies, 0.95), 6)
            if successful_latencies
            else None,
        },
        "ann_fidelity": {
            "metric": f"set_overlap@{config.top_k}",
            "scope": "ANN versus exact vector retrieval; not recommendation quality",
            "successful_queries_only": True,
            "mean_overlap_at_k": sum(overlaps) / len(overlaps) if overlaps else None,
        },
        "privacy": {
            "query_identifiers": "omitted",
            "query_embeddings": "omitted",
            "returned_item_identifiers": "omitted",
        },
        "samples": samples,
    }
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    """Validate both JSON shape and cross-field measurement invariants."""

    _validate_json_schema(report, "result.schema.json", "runtime report")
    samples = report["samples"]
    warmups = [sample for sample in samples if sample["phase"] == "warmup"]
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    workload = report["workload"]
    if [sample["sequence"] for sample in warmups] != list(range(len(warmups))):
        raise BenchmarkContractError("warmup sample sequence is not contiguous")
    if [sample["sequence"] for sample in measured] != list(range(len(measured))):
        raise BenchmarkContractError("measured sample sequence is not contiguous")
    warmup_failures = sum(not sample["success"] for sample in warmups)
    measured_successes = [sample for sample in measured if sample["success"]]
    measured_failures = len(measured) - len(measured_successes)
    expected_counts = {
        "warmups_requested": len(warmups),
        "warmups_success": len(warmups) - warmup_failures,
        "warmups_failure": warmup_failures,
        "queries_requested": len(measured),
        "queries_success": len(measured_successes),
        "queries_failure": measured_failures,
    }
    if any(workload[key] != value for key, value in expected_counts.items()):
        raise BenchmarkContractError("runtime report counters do not match samples")
    for sample in samples:
        if sample["success"]:
            if (
                sample["error"] is not None
                or sample["exact_ann_overlap_count"] is None
                or sample["exact_ann_overlap_at_k"] is None
            ):
                raise BenchmarkContractError("successful sample fields are inconsistent")
            expected_ratio = sample["exact_ann_overlap_count"] / workload["top_k"]
            if sample["exact_ann_overlap_at_k"] != expected_ratio:
                raise BenchmarkContractError("sample overlap count and ratio differ")
        elif (
            sample["error"] is None
            or sample["exact_ann_overlap_count"] is not None
            or sample["exact_ann_overlap_at_k"] is not None
        ):
            raise BenchmarkContractError("failed sample fields are inconsistent")
    latencies = [sample["latency_ms"] for sample in measured_successes]
    overlaps = [sample["exact_ann_overlap_at_k"] for sample in measured_successes]
    expected_p50 = round(_percentile(latencies, 0.50), 6) if latencies else None
    expected_p95 = round(_percentile(latencies, 0.95), 6) if latencies else None
    expected_overlap = sum(overlaps) / len(overlaps) if overlaps else None
    if report["latency_ms"]["p50"] != expected_p50 or report["latency_ms"]["p95"] != expected_p95:
        raise BenchmarkContractError("runtime report latency summaries are inconsistent")
    if report["ann_fidelity"]["mean_overlap_at_k"] != expected_overlap:
        raise BenchmarkContractError("runtime report overlap summary is inconsistent")
    if expected_p50 is not None and expected_p95 < expected_p50:
        raise BenchmarkContractError("runtime report p95 is below p50")
    before = report["control_plane"]["before"]
    after = report["control_plane"]["after"]
    unchanged = report["control_plane"]["unchanged"]
    controls_match = after is not None and _control_identity(before) == _control_identity(after)
    all_calls_succeeded = warmup_failures == 0 and measured_failures == 0
    should_pass = all_calls_succeeded and controls_match and unchanged is True
    if (report["status"] == "passed") != should_pass:
        raise BenchmarkContractError("runtime report pass/fail status is inconsistent")
    if (report["failure"] is None) != (report["status"] == "passed"):
        raise BenchmarkContractError("runtime report failure record is inconsistent")
    expected_pointer = report["input"]["expected_active_pointer"]
    if before["active_pointer"] != expected_pointer:
        raise BenchmarkContractError("preflight pointer does not match the input")
    if before["active_pointer_sha256"] != _canonical_sha256(before["active_pointer"]):
        raise BenchmarkContractError("preflight pointer digest is inconsistent")
    if after is not None and after["active_pointer_sha256"] != _canonical_sha256(
        after["active_pointer"]
    ):
        raise BenchmarkContractError("postflight pointer digest is inconsistent")
    manifest_contract = report["input"]["manifest_contract"]
    manifest_evidence = before["promotion_manifest"]
    if (
        manifest_evidence["identity"] != expected_pointer["manifest"]
        or manifest_evidence["generation_id"] != expected_pointer["generation_id"]
        or manifest_evidence["dimensions"] != manifest_contract["dimensions"]
        or manifest_evidence["provenance"] != expected_pointer["provenance"]
        or any(
            manifest_evidence["items"][key] != manifest_contract["items"][key]
            for key in (
                "count",
                "sha256",
                "canonical_uri",
                "canonical_object_generation",
            )
        )
    ):
        raise BenchmarkContractError("preflight manifest evidence is inconsistent")
    vertex = before["vertex"]
    if (
        any(
            vertex[key] != expected_pointer["serving"][key]
            for key in ("index_resource", "index_endpoint", "deployed_index_id")
        )
        or vertex["dimensions"] != manifest_contract["dimensions"]
        or vertex["distance_metric"] != workload["distance_metric"]
        or vertex["index_update_method"] != "STREAM_UPDATE"
        or vertex["deployed_mapping_verified"] is not True
    ):
        raise BenchmarkContractError("preflight Vertex evidence is inconsistent")
    if workload["catalog_canonical_sha256"] != report["input"]["manifest_contract"]["items"]["sha256"]:
        raise BenchmarkContractError("report catalog hash does not match the manifest")
    plan = report["plan"]
    calls = plan["planned_cloud_api_calls"]
    expected_find_calls = workload["warmups_requested"] + workload["queries_requested"]
    expected_total_calls = FIXED_CONTROL_PLANE_CALLS + expected_find_calls
    expected_phrase = LIVE_CONFIRMATION_TEMPLATE.format(
        calls=expected_total_calls,
        generation=expected_pointer["generation_id"],
        epoch=expected_pointer["epoch"],
    )
    if (
        calls["vertex_find_neighbors_calls"] != expected_find_calls
        or calls["total_potentially_billable_calls"] != expected_total_calls
        or plan["planned_exact_scalar_operations"]
        != workload["queries_requested"]
        * workload["catalog_size"]
        * workload["dimensions"]
        or plan["required_confirmation_phrase"] != expected_phrase
    ):
        raise BenchmarkContractError("runtime report plan is inconsistent")
    attestations = report["operator_attestations"]
    if (
        attestations["source_provenance"]["value"]
        != report["input"]["source"]["description"]
        or attestations["client_region"]["verification"] != OPERATOR_ATTESTED
        or attestations["source_provenance"]["verification"] != OPERATOR_ATTESTED
    ):
        raise BenchmarkContractError("runtime report attestations are inconsistent")
    if report["ann_fidelity"]["metric"] != f"set_overlap@{workload['top_k']}":
        raise BenchmarkContractError("runtime report fidelity metric is inconsistent")
    if any(key == "query_id" for key in _walk_keys(report)):
        raise BenchmarkContractError("runtime report contains a raw query identifier")


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def repository_commit_and_clean() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchmarkContractError("live benchmark must run from a Git checkout") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise BenchmarkContractError("could not resolve an exact repository commit")
    if dirty:
        raise BenchmarkContractError(
            "live benchmark evidence requires a clean Git worktree; commit changes first"
        )
    return commit


def credentialed_cloud(binding: ServingBinding) -> CloudClients:
    """Build read/query clients after explicitly refreshing ADC credentials."""

    try:
        import google.auth
        from google.api_core.client_options import ClientOptions
        from google.auth.transport.requests import Request
        from google.cloud import aiplatform, aiplatform_v1, bigquery, storage
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
            Namespace,
        )
    except ImportError as exc:
        raise BenchmarkContractError(
            "install benchmarks/vector_search_runtime/requirements.txt first"
        ) from exc
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
    except Exception as exc:
        raise BenchmarkContractError("ADC credentials could not be validated") from exc
    if not credentials.valid:
        raise BenchmarkContractError("ADC credentials could not be validated")
    project = binding.index_name.split("/")[1]
    location = binding.index_name.split("/")[3]
    api_options = ClientOptions(api_endpoint=f"{location}-aiplatform.googleapis.com")
    aiplatform.init(project=project, location=location, credentials=credentials)
    try:
        query_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=binding.index_endpoint,
            credentials=credentials,
        )
        index_service = aiplatform_v1.IndexServiceClient(
            credentials=credentials, client_options=api_options
        )
        endpoint_service = aiplatform_v1.IndexEndpointServiceClient(
            credentials=credentials, client_options=api_options
        )
    except Exception as exc:
        raise BenchmarkContractError("required existing Vertex resources could not be loaded") from exc
    return CloudClients(
        bigquery=bigquery.Client(project=project, credentials=credentials),
        storage=storage.Client(project=project, credentials=credentials),
        index_service=index_service,
        endpoint_service=endpoint_service,
        query_endpoint=query_endpoint,
        namespace_factory=Namespace,
        sdk_version=importlib.metadata.version("google-cloud-aiplatform"),
    )


def _diagnostic(
    *,
    status: str,
    stage: str,
    error: Exception | None,
    plan: Mapping[str, Any] | None,
    artifact_sha256: str | None,
) -> dict[str, Any]:
    document = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_kind": DIAGNOSTIC_KIND,
        "status": status,
        "stage": stage,
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "error": _error_record(error, stage) if error is not None else None,
        "input_artifact_sha256": artifact_sha256,
        "plan": dict(plan) if plan is not None else None,
    }
    _validate_json_schema(document, "diagnostic.schema.json", "diagnostic report")
    return document


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Write a private report by fsync + atomic replace in the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-plan or benchmark one active Vertex generation without mutating cloud resources."
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-pointer-table", required=True)
    parser.add_argument("--index-resource", required=True)
    parser.add_argument("--index-endpoint", required=True)
    parser.add_argument("--deployed-index-id", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--promotion-epoch", type=int, required=True)
    parser.add_argument("--client-region", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument(
        "--attest-source-is-production",
        action="store_true",
        help="operator attestation only; the harness cannot independently verify provenance",
    )
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="make the planned read/query cloud calls; omitted means no-cloud dry run",
    )
    parser.add_argument(
        "--confirmation",
        help="exact dynamic phrase emitted by the preceding dry run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    stage = "input_validation"
    plan: Mapping[str, Any] | None = None
    artifact_sha256: str | None = None
    try:
        if args.artifact.resolve() == args.output.resolve():
            raise BenchmarkContractError("output must not overwrite the private input")
        artifact, artifact_sha256 = load_artifact(args.artifact)
        try:
            binding = validate_binding(
                args.index_resource, args.index_endpoint, args.deployed_index_id
            )
            generation_id = validate_generation_id(args.generation_id)
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError(str(exc)) from exc
        config = BenchmarkConfig(
            binding=binding,
            active_pointer_table=args.active_pointer_table,
            generation_id=generation_id,
            promotion_epoch=args.promotion_epoch,
            client_region=args.client_region,
            top_k=args.top_k,
            warmups=args.warmups,
            query_limit=args.query_limit,
            artifact_sha256=artifact_sha256,
            repository_commit=None,
        )
        validate_config(config, artifact)
        plan = build_plan(config, artifact)
        if not args.execute_live:
            diagnostic = _diagnostic(
                status="planned",
                stage="dry_run",
                error=None,
                plan=plan,
                artifact_sha256=artifact_sha256,
            )
            write_json_atomic(args.output, diagnostic)
            print(json.dumps(plan, indent=2, sort_keys=True))
            print(f"dry-run plan written to {args.output}; no cloud calls were made")
            return 0
        stage = "authorization"
        if not args.attest_source_is_production:
            raise BenchmarkContractError(
                "--attest-source-is-production is required for live execution"
            )
        if args.confirmation != plan["required_confirmation_phrase"]:
            raise BenchmarkContractError("live confirmation phrase is absent or does not match")
        stage = "repository_binding"
        commit = repository_commit_and_clean()
        config = BenchmarkConfig(**{**config.__dict__, "repository_commit": commit})
        stage = "control_preflight"
        cloud = credentialed_cloud(binding)
        report = run_benchmark(
            config=config,
            artifact=artifact,
            endpoint=cloud.query_endpoint,
            namespace_factory=cloud.namespace_factory,
            sdk_version=cloud.sdk_version,
            control_reader=lambda: capture_control_snapshot(cloud, artifact),
        )
        stage = "report_validation"
        validate_report(report)
        write_json_atomic(args.output, report)
        if report["status"] != "passed":
            print(
                f"benchmark failed closed; diagnostic result written to {args.output}",
                file=sys.stderr,
            )
            return 2
        print(f"benchmark report written to {args.output}")
        return 0
    except Exception as exc:
        try:
            diagnostic = _diagnostic(
                status="refused",
                stage=stage,
                error=exc,
                plan=plan,
                artifact_sha256=artifact_sha256,
            )
            write_json_atomic(args.output, diagnostic)
        except Exception as report_error:
            print(
                f"benchmark refused to run and diagnostic write failed: {type(report_error).__name__}",
                file=sys.stderr,
            )
        print(f"benchmark refused to run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

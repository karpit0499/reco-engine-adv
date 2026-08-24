"""Side-effect-free contracts for generation-filtered embedding promotion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 3
GENERATION_NAMESPACE = "generation"
GENERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
INDEX_RE = re.compile(r"^projects/[^/]+/locations/[^/]+/indexes/[^/]+$")
ENDPOINT_RE = re.compile(
    r"^projects/[^/]+/locations/[^/]+/indexEndpoints/[^/]+$"
)
DEPLOYED_INDEX_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
OPS_IMAGE_RE = re.compile(r"^[^\s:@]+(?:/[^\s:@]+)+@sha256:[0-9a-f]{64}$")
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CANDIDATE_PREFIX = "candidate-generations"
USER_GENERATIONS_TABLE = "user_embedding_generations"
ACTIVE_GENERATION_TABLE = "active_embedding_generation"
GENERATION_RETIREMENTS_TABLE = "embedding_generation_retirements"
INITIALIZATION_LOCK_BLOB = "promotion-control/initialization.json"
DATAPOINT_SEPARATOR = "::"


@dataclass(frozen=True)
class ServingBinding:
    index_name: str
    index_endpoint: str
    deployed_index_id: str


@dataclass(frozen=True)
class ManifestReceipt:
    uri: str
    object_generation: int
    sha256: str


def validate_generation_id(value: str) -> str:
    if not isinstance(value, str) or not GENERATION_ID_RE.fullmatch(value):
        raise ValueError(
            "generation_id must be 1-64 safe characters and start with a letter or number"
        )
    return value


def validate_binding(
    index_name: str, index_endpoint: str, deployed_index_id: str
) -> ServingBinding:
    if not isinstance(index_name, str) or not INDEX_RE.fullmatch(index_name):
        raise ValueError("index_name must be a fully-qualified Vertex index resource")
    if not isinstance(index_endpoint, str) or not ENDPOINT_RE.fullmatch(index_endpoint):
        raise ValueError(
            "index_endpoint must be a fully-qualified Vertex index endpoint resource"
        )
    if (
        not isinstance(deployed_index_id, str)
        or not DEPLOYED_INDEX_ID_RE.fullmatch(deployed_index_id)
    ):
        raise ValueError("deployed_index_id has an invalid format")
    index_location = index_name.split("/")[3]
    endpoint_location = index_endpoint.split("/")[3]
    if index_location != endpoint_location:
        raise ValueError("index and endpoint must be in the same location")
    return ServingBinding(index_name, index_endpoint, deployed_index_id)


def validate_ops_image_uri(value: str) -> str:
    if not isinstance(value, str) or not OPS_IMAGE_RE.fullmatch(value):
        raise ValueError("ops image must be an OCI URI pinned by sha256 digest")
    return value


def validate_source_revision(value: str) -> str:
    if not isinstance(value, str) or not SOURCE_REVISION_RE.fullmatch(value):
        raise ValueError("source revision must be a full lowercase Git commit")
    return value


def validate_sha256(value: str, field: str = "sha256") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def item_blob_name(generation_id: str) -> str:
    return f"{CANDIDATE_PREFIX}/{validate_generation_id(generation_id)}/items.json"


def user_blob_name(generation_id: str) -> str:
    return f"{CANDIDATE_PREFIX}/{validate_generation_id(generation_id)}/users.json"


def manifest_blob_name(generation_id: str) -> str:
    return f"{CANDIDATE_PREFIX}/{validate_generation_id(generation_id)}/manifest.json"


def manifest_uri(bucket: str, generation_id: str) -> str:
    return f"gs://{bucket}/{manifest_blob_name(generation_id)}"


def namespaced_datapoint_id(generation_id: str, item_id: str) -> str:
    generation_id = validate_generation_id(generation_id)
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("item_id must be a non-empty string")
    return f"{generation_id}{DATAPOINT_SEPARATOR}{item_id}"


def strip_namespaced_datapoint_id(generation_id: str, datapoint_id: str) -> str:
    prefix = f"{validate_generation_id(generation_id)}{DATAPOINT_SEPARATOR}"
    if not isinstance(datapoint_id, str) or not datapoint_id.startswith(prefix):
        raise ValueError("Vertex returned a datapoint outside the requested generation")
    item_id = datapoint_id[len(prefix) :]
    if not item_id:
        raise ValueError("Vertex returned an empty item id")
    return item_id


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def active_state_receipt_sha256(
    *,
    generation_id: str,
    epoch: int,
    manifest_uri_value: str,
    manifest_object_generation: int,
    manifest_sha256: str,
    ops_image_uri: str,
    source_revision: str,
    champion_score: float,
    binding: ServingBinding,
) -> str:
    """Hash the complete immutable state used by an evaluation decision.

    The hash is the cross-step evaluation fence. Promotion still compares every
    underlying field in BigQuery; the digest keeps the KFP hand-off compact while
    binding the champion score, provenance, manifest version, and serving target.
    """

    generation_id = validate_generation_id(generation_id)
    if not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("active epoch must be a positive integer")
    if manifest_uri_value != manifest_uri(
        manifest_uri_value.removeprefix("gs://").split("/", 1)[0], generation_id
    ):
        raise ValueError("active manifest URI is not generation-bound")
    if not isinstance(manifest_object_generation, int) or manifest_object_generation <= 0:
        raise ValueError("active manifest object generation is invalid")
    manifest_sha256 = validate_sha256(manifest_sha256, "active manifest digest")
    ops_image_uri = validate_ops_image_uri(ops_image_uri)
    source_revision = validate_source_revision(source_revision)
    if (
        not isinstance(champion_score, (int, float))
        or not math.isfinite(float(champion_score))
        or not 0.0 <= float(champion_score) <= 1.0
    ):
        raise ValueError("active champion score is invalid")
    binding = validate_binding(
        binding.index_name, binding.index_endpoint, binding.deployed_index_id
    )
    payload = {
        "binding": {
            "deployed_index_id": binding.deployed_index_id,
            "index_endpoint": binding.index_endpoint,
            "index_name": binding.index_name,
        },
        "champion_score": float(champion_score),
        "epoch": epoch,
        "generation_id": generation_id,
        "manifest_object_generation": manifest_object_generation,
        "manifest_sha256": manifest_sha256,
        "manifest_uri": manifest_uri_value,
        "ops_image_uri": ops_image_uri,
        "source_revision": source_revision,
    }
    return sha256_bytes(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())


def _finite_vector(value: Any, *, dimensions: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != dimensions:
        raise ValueError(f"embedding must contain exactly {dimensions} values")
    vector = [float(component) for component in value]
    if not all(math.isfinite(component) for component in vector):
        raise ValueError("embedding values must be finite")
    return vector


def canonical_item_payload(
    rows: Iterable[Mapping[str, Any]], *, dimensions: int
) -> tuple[bytes, list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("every item needs a non-empty string id")
        normalized.append(
            {
                "id": item_id,
                "embedding": _finite_vector(row.get("embedding"), dimensions=dimensions),
            }
        )
    normalized.sort(key=lambda row: row["id"])
    if not normalized or len({row["id"] for row in normalized}) != len(normalized):
        raise ValueError("item generation must be non-empty with unique ids")
    payload = (
        "\n".join(json.dumps(row, separators=(",", ":")) for row in normalized) + "\n"
    ).encode()
    return payload, normalized


def canonical_user_payload(
    rows: Iterable[Mapping[str, Any]], *, dimensions: int
) -> tuple[bytes, list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        user_id = row.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("every user needs a non-empty string user_id")
        normalized.append(
            {
                "user_id": user_id,
                "embedding": _finite_vector(row.get("embedding"), dimensions=dimensions),
            }
        )
    normalized.sort(key=lambda row: row["user_id"])
    if not normalized or len({row["user_id"] for row in normalized}) != len(normalized):
        raise ValueError("user generation must be non-empty with unique ids")
    payload = (
        "\n".join(json.dumps(row, separators=(",", ":")) for row in normalized) + "\n"
    ).encode()
    return payload, normalized


def parse_item_payload(payload: bytes, *, dimensions: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    canonical, normalized = canonical_item_payload(rows, dimensions=dimensions)
    if canonical != payload:
        raise ValueError("item artifact is not canonical JSONL")
    return normalized


def parse_user_payload(payload: bytes, *, dimensions: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    canonical, normalized = canonical_user_payload(rows, dimensions=dimensions)
    if canonical != payload:
        raise ValueError("user artifact is not canonical JSONL")
    return normalized


def build_manifest(
    *,
    generation_id: str,
    created_at: str,
    trained_at: str,
    dimensions: int,
    items_payload: bytes,
    item_count: int,
    canonical_bucket: str,
    canonical_item_generation: int,
    regional_bucket: str,
    regional_item_generation: int,
    users_payload: bytes,
    user_count: int,
    user_object_generation: int,
    model_uri: str,
    project: str,
    binding: ServingBinding,
    ops_image_uri: str,
    source_revision: str,
    canary_item_id: str,
    canary_user_id: str,
) -> dict[str, Any]:
    generation_id = validate_generation_id(generation_id)
    binding = validate_binding(
        binding.index_name, binding.index_endpoint, binding.deployed_index_id
    )
    ops_image_uri = validate_ops_image_uri(ops_image_uri)
    source_revision = validate_source_revision(source_revision)
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_id": generation_id,
        "created_at": created_at,
        "trained_at": trained_at,
        "dimensions": dimensions,
        "serving": {
            "index_name": binding.index_name,
            "index_endpoint": binding.index_endpoint,
            "deployed_index_id": binding.deployed_index_id,
            "generation_namespace": GENERATION_NAMESPACE,
        },
        "provenance": {
            "ops_image_uri": ops_image_uri,
            "source_revision": source_revision,
        },
        "items": {
            "count": item_count,
            "sha256": sha256_bytes(items_payload),
            "canonical_uri": f"gs://{canonical_bucket}/{item_blob_name(generation_id)}",
            "canonical_object_generation": canonical_item_generation,
            "regional_uri": f"gs://{regional_bucket}/{item_blob_name(generation_id)}",
            "regional_object_generation": regional_item_generation,
            "canary_item_id": canary_item_id,
            "canary_datapoint_id": namespaced_datapoint_id(
                generation_id, canary_item_id
            ),
        },
        "users": {
            "count": user_count,
            "dimensions": dimensions,
            "sha256": sha256_bytes(users_payload),
            "uri": f"gs://{canonical_bucket}/{user_blob_name(generation_id)}",
            "object_generation": user_object_generation,
            "table": f"{project}.reco.{USER_GENERATIONS_TABLE}",
            "generation_id": generation_id,
            "canary_user_id": canary_user_id,
        },
        "models": {
            "user_tower_uri": f"{model_uri.rstrip('/')}/user_tower",
            "item_tower_uri": f"{model_uri.rstrip('/')}/item_tower",
        },
    }


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_generation_id: str,
    canonical_bucket: str,
    regional_bucket: str,
    expected_binding: ServingBinding,
    expected_ops_image_uri: str,
    expected_source_revision: str,
) -> dict[str, Any]:
    generation_id = validate_generation_id(expected_generation_id)
    expected_binding = validate_binding(
        expected_binding.index_name,
        expected_binding.index_endpoint,
        expected_binding.deployed_index_id,
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("generation_id") != generation_id:
        raise ValueError("manifest generation does not match the requested generation")
    dimensions = manifest.get("dimensions")
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("manifest dimensions must be positive")

    serving = manifest.get("serving") or {}
    actual_binding = validate_binding(
        serving.get("index_name"),
        serving.get("index_endpoint"),
        serving.get("deployed_index_id"),
    )
    if actual_binding != expected_binding:
        raise ValueError("manifest is bound to a different Vertex serving target")
    if serving.get("generation_namespace") != GENERATION_NAMESPACE:
        raise ValueError("manifest generation namespace is invalid")

    provenance = manifest.get("provenance") or {}
    actual_image = validate_ops_image_uri(provenance.get("ops_image_uri"))
    actual_revision = validate_source_revision(provenance.get("source_revision"))
    if actual_image != validate_ops_image_uri(expected_ops_image_uri):
        raise ValueError("manifest was produced by a different ops image")
    if actual_revision != validate_source_revision(expected_source_revision):
        raise ValueError("manifest was produced from a different source revision")

    items = manifest.get("items") or {}
    users = manifest.get("users") or {}
    expected_item_blob = item_blob_name(generation_id)
    expected_user_blob = user_blob_name(generation_id)
    if items.get("canonical_uri") != f"gs://{canonical_bucket}/{expected_item_blob}":
        raise ValueError("manifest canonical item URI is invalid")
    if items.get("regional_uri") != f"gs://{regional_bucket}/{expected_item_blob}":
        raise ValueError("manifest regional item URI is invalid")
    if users.get("uri") != f"gs://{canonical_bucket}/{expected_user_blob}":
        raise ValueError("manifest user URI is invalid")
    for section, field in (
        (items, "canonical_object_generation"),
        (items, "regional_object_generation"),
        (users, "object_generation"),
    ):
        if not isinstance(section.get(field), int) or section[field] <= 0:
            raise ValueError(f"manifest {field} is invalid")
    for section in (items, users):
        if not isinstance(section.get("count"), int) or section["count"] <= 0:
            raise ValueError("manifest counts must be positive")
        validate_sha256(section.get("sha256"), "manifest artifact digest")
    if users.get("dimensions") != dimensions or users.get("generation_id") != generation_id:
        raise ValueError("manifest user generation/dimensions are inconsistent")
    canary_item_id = items.get("canary_item_id")
    if not isinstance(canary_item_id, str) or not canary_item_id:
        raise ValueError("manifest item canary is invalid")
    if items.get("canary_datapoint_id") != namespaced_datapoint_id(
        generation_id, canary_item_id
    ):
        raise ValueError("manifest canary datapoint is invalid")
    if not isinstance(users.get("canary_user_id"), str) or not users["canary_user_id"]:
        raise ValueError("manifest user canary is invalid")
    return dict(manifest)


def promotion_decision(
    *, score: float, floor: float, margin: float, champion: float | None
) -> tuple[str, str]:
    if not 0.0 <= score <= 1.0 or not 0.0 <= floor <= 1.0 or margin < 0.0:
        raise ValueError("gate inputs are outside their valid range")
    if score < floor:
        return "hold", f"{score:.1%} is below the {floor:.0%} floor"
    if champion is not None and score < champion - margin:
        return (
            "hold",
            f"{score:.1%} regressed more than {margin:.0%} below champion {champion:.1%}",
        )
    return "promote", "no material regression and above the floor"

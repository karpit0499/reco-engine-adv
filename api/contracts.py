"""Pure request-contract and bearer-authentication helpers for the API."""

from __future__ import annotations

import hmac
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MAX_JSON_BYTES = 2_048
MAX_USER_ID_LENGTH = 128
MIN_BEARER_TOKEN_LENGTH = 32
MAX_BEARER_TOKEN_LENGTH = 512

_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,127}$")
_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_INDEX_RESOURCE = re.compile(r"^projects/[^/]+/locations/[^/]+/indexes/[^/]+$")
_INDEX_ENDPOINT = re.compile(
    r"^projects/[^/]+/locations/[^/]+/indexEndpoints/[^/]+$"
)
_DEPLOYED_INDEX_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPS_IMAGE = re.compile(r"^[^\s:@]+(?:/[^\s:@]+)+@sha256:[0-9a-f]{64}$")
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
DATAPOINT_SEPARATOR = "::"


class ContractError(ValueError):
    """Raised when a request does not match its exact public contract."""


@dataclass(frozen=True)
class RecommendRequest:
    user_id: str
    request_id: str


@dataclass(frozen=True)
class FeedbackRequest:
    user_id: str
    impression_id: str
    event_id: str
    event: str


@dataclass(frozen=True)
class ServingTarget:
    index_name: str
    index_endpoint: str
    deployed_index_id: str


@dataclass(frozen=True)
class ServingVector:
    embedding: tuple[float, ...]
    generation_id: str
    epoch: int
    target: ServingTarget


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _resource(client: Any) -> Any:
    return getattr(client, "gca_resource", client)


def _resource_name(client: Any) -> str | None:
    name = getattr(client, "resource_name", None)
    if name:
        return str(name)
    resource = _resource(client)
    name = _field(resource, "name")
    return str(name) if name else None


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def verify_vertex_target(
    index: Any,
    endpoint: Any,
    *,
    target: ServingTarget,
    dimensions: int,
) -> None:
    """Fail closed unless the live Vertex resources match the serving contract.

    A deployed-index id is mutable control-plane state. Checking only its string
    value in BigQuery is insufficient because it can later be redeployed to a
    different index. Callers therefore construct fresh clients and verify the
    live mapping immediately before every treatment query.
    """

    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ContractError("serving embedding dimension is invalid")
    if _resource_name(index) != target.index_name:
        raise ContractError("live Vertex index resource differs from the active binding")
    if _resource_name(endpoint) != target.index_endpoint:
        raise ContractError("live Vertex endpoint resource differs from the active binding")

    endpoint_resource = _resource(endpoint)
    deployments = _field(endpoint_resource, "deployed_indexes")
    if deployments is None:
        deployments = getattr(endpoint, "deployed_indexes", None)
    matches = [
        deployment
        for deployment in list(deployments or [])
        if str(_field(deployment, "id", "")) == target.deployed_index_id
    ]
    if len(matches) != 1 or str(_field(matches[0], "index", "")) != target.index_name:
        raise ContractError("deployed Vertex id is not mapped to the active index")

    index_resource = _resource(index)
    metadata = _field(index_resource, "metadata") or {}
    config = _field(metadata, "config") or {}
    live_dimensions = _field(config, "dimensions")
    try:
        live_dimensions_number = int(live_dimensions)
    except (TypeError, ValueError):
        raise ContractError("live Vertex index dimension is unavailable") from None
    if live_dimensions_number != dimensions or float(live_dimensions) != dimensions:
        raise ContractError("live Vertex index dimension differs from the active model")

    distance = _field(config, "distanceMeasureType")
    if distance is None:
        distance = _field(config, "distance_measure_type")
    if _enum_name(distance) != "DOT_PRODUCT_DISTANCE":
        raise ContractError("live Vertex index must use DOT_PRODUCT_DISTANCE")

    update_method = _field(index_resource, "index_update_method")
    if _enum_name(update_method) != "STREAM_UPDATE":
        raise ContractError("live Vertex index must use STREAM_UPDATE")


def serving_target(
    index_name: str | None,
    index_endpoint: str | None,
    deployed_index_id: str | None,
) -> ServingTarget:
    if not isinstance(index_name, str) or not _INDEX_RESOURCE.fullmatch(index_name):
        raise ContractError("INDEX_RESOURCE_NAME is not a full Vertex index resource")
    if not isinstance(index_endpoint, str) or not _INDEX_ENDPOINT.fullmatch(index_endpoint):
        raise ContractError("INDEX_ENDPOINT is not a full Vertex index endpoint resource")
    if (
        not isinstance(deployed_index_id, str)
        or not _DEPLOYED_INDEX_ID.fullmatch(deployed_index_id)
    ):
        raise ContractError("DEPLOYED_INDEX_ID is invalid")
    if index_name.split("/")[3] != index_endpoint.split("/")[3]:
        raise ContractError("Vertex index and endpoint locations differ")
    return ServingTarget(index_name, index_endpoint, deployed_index_id)


def validate_generation_id(value: str) -> str:
    if not isinstance(value, str) or not _GENERATION_ID.fullmatch(value):
        raise ContractError("active generation id is invalid")
    return value


def validate_active_receipt(
    generation_id: str,
    *,
    manifest_uri: Any,
    manifest_object_generation: Any,
    manifest_sha256: Any,
    ops_image_uri: Any,
    source_revision: Any,
) -> None:
    generation_id = validate_generation_id(generation_id)
    if not isinstance(manifest_uri, str) or not re.fullmatch(
        rf"gs://[a-z0-9][a-z0-9._-]*/candidate-generations/{re.escape(generation_id)}/manifest\.json",
        manifest_uri,
    ):
        raise ContractError("active manifest URI is malformed")
    if (
        not isinstance(manifest_object_generation, int)
        or isinstance(manifest_object_generation, bool)
        or manifest_object_generation <= 0
    ):
        raise ContractError("active manifest object generation is invalid")
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        raise ContractError("active manifest digest is invalid")
    if not isinstance(ops_image_uri, str) or not _OPS_IMAGE.fullmatch(ops_image_uri):
        raise ContractError("active ops image is not digest-pinned")
    if not isinstance(source_revision, str) or not _SOURCE_REVISION.fullmatch(source_revision):
        raise ContractError("active source revision is invalid")


def strip_generation_datapoint(generation_id: str, datapoint_id: str) -> str:
    generation_id = validate_generation_id(generation_id)
    prefix = f"{generation_id}{DATAPOINT_SEPARATOR}"
    if not isinstance(datapoint_id, str) or not datapoint_id.startswith(prefix):
        raise ContractError("Vector Search returned a mixed generation")
    item_id = datapoint_id[len(prefix) :]
    if not item_id:
        raise ContractError("Vector Search returned an empty item id")
    return item_id


def bearer_auth_status(configured_token: str | None, authorization: str | None) -> str:
    """Return ``ok``, ``unauthorized``, or ``misconfigured`` without secrets.

    A missing, weak, oversized, or whitespace-containing configured token is a
    server configuration failure. It must never turn into an authentication
    bypass.
    """

    if (
        not isinstance(configured_token, str)
        or not MIN_BEARER_TOKEN_LENGTH <= len(configured_token) <= MAX_BEARER_TOKEN_LENGTH
        or any(character.isspace() for character in configured_token)
    ):
        return "misconfigured"

    if not isinstance(authorization, str) or len(authorization) > MAX_BEARER_TOKEN_LENGTH + 7:
        return "unauthorized"

    scheme, separator, supplied_token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not supplied_token:
        return "unauthorized"
    if any(character.isspace() for character in supplied_token):
        return "unauthorized"

    return "ok" if hmac.compare_digest(configured_token, supplied_token) else "unauthorized"


def parse_recommend(body: Any) -> RecommendRequest:
    obj = _exact_object(body, {"user_id", "request_id"})
    return RecommendRequest(
        user_id=_user_id(obj["user_id"]),
        request_id=_canonical_uuid(obj["request_id"], "request_id"),
    )


def parse_feedback(body: Any) -> FeedbackRequest:
    obj = _exact_object(body, {"user_id", "impression_id", "event_id", "event"})
    event = obj["event"]
    if not isinstance(event, str) or event not in {"click", "conversion"}:
        raise ContractError("event must be click or conversion")
    return FeedbackRequest(
        user_id=_user_id(obj["user_id"]),
        impression_id=_canonical_uuid(obj["impression_id"], "impression_id"),
        event_id=_canonical_uuid(obj["event_id"], "event_id"),
        event=event,
    )


def _exact_object(body: Any, required_keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(body, dict):
        raise ContractError("request body must be a JSON object")
    if set(body) != required_keys:
        raise ContractError("request body has missing or unexpected fields")
    return body


def _user_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_USER_ID_LENGTH
        or not _USER_ID.fullmatch(value)
    ):
        raise ContractError("user_id has an invalid format")
    return value


def _canonical_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 36:
        raise ContractError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        raise ContractError(f"{field} must be a canonical UUID") from None
    if str(parsed) != value or parsed.int == 0 or parsed.variant != uuid.RFC_4122:
        raise ContractError(f"{field} must be a canonical UUID")
    return value

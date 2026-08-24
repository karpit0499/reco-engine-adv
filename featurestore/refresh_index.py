"""Generation-filtered Vertex Vector Search propagation helpers.

Candidate generations coexist in one streaming index. Serving requests select
exactly one generation namespace, so propagation never rewrites the active set.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any


GENERATION_NAMESPACE = "generation"


def namespaced_id(generation_id: str, item_id: str) -> str:
    if not generation_id or not item_id:
        raise ValueError("generation and item ids must be non-empty")
    return f"{generation_id}::{item_id}"


def datapoint_specs(
    rows: Sequence[dict[str, Any]], generation_id: str
) -> list[dict[str, Any]]:
    specs = []
    seen = set()
    dimensions = None
    for row in rows:
        item_id = row.get("id")
        vector = row.get("embedding")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("candidate items need unique, non-empty string ids")
        if not isinstance(vector, list) or not vector:
            raise ValueError("candidate item vectors must be non-empty lists")
        dimensions = len(vector) if dimensions is None else dimensions
        if len(vector) != dimensions or not all(math.isfinite(float(v)) for v in vector):
            raise ValueError("candidate item vectors have inconsistent or invalid dimensions")
        seen.add(item_id)
        specs.append(
            {
                "datapoint_id": namespaced_id(generation_id, item_id),
                "feature_vector": [float(value) for value in vector],
                "restricts": [
                    {
                        "namespace": GENERATION_NAMESPACE,
                        "allow_list": [generation_id],
                    }
                ],
            }
        )
    if not specs:
        raise ValueError("candidate generation is empty")
    return specs


def _vertex_datapoints(specs: Sequence[dict[str, Any]]):
    from google.cloud import aiplatform_v1

    datapoint_type = aiplatform_v1.types.IndexDatapoint
    restriction_type = datapoint_type.Restriction
    return [
        datapoint_type(
            datapoint_id=spec["datapoint_id"],
            feature_vector=spec["feature_vector"],
            restricts=[
                restriction_type(
                    namespace=restriction["namespace"],
                    allow_list=restriction["allow_list"],
                )
                for restriction in spec["restricts"]
            ],
        )
        for spec in specs
    ]


def _batches(values: Sequence[Any], size: int):
    if size <= 0:
        raise ValueError("batch size must be positive")
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def upsert_generation(
    index: Any,
    *,
    rows: Sequence[dict[str, Any]],
    generation_id: str,
    fence: Callable[[], None],
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    """Upsert namespaced datapoints alongside older generations; delete none."""
    specs = datapoint_specs(rows, generation_id)
    datapoints = _vertex_datapoints(specs)
    for batch in _batches(datapoints, batch_size):
        fence()
        index.upsert_datapoints(datapoints=batch)
    return specs


def _restriction_tokens(datapoint: Any, namespace: str) -> set[str]:
    tokens: set[str] = set()
    for restriction in getattr(datapoint, "restricts", []):
        if getattr(restriction, "namespace", None) == namespace:
            tokens.update(getattr(restriction, "allow_list", []))
    return tokens


def _vector_matches(actual: Sequence[float], expected: Sequence[float]) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7)
        for left, right in zip(actual, expected)
    )


def _resource_name(resource: Any) -> str | None:
    for candidate in (
        getattr(resource, "resource_name", None),
        getattr(resource, "name", None),
        getattr(getattr(resource, "gca_resource", None), "name", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _metadata_dict(index: Any) -> dict[str, Any]:
    metadata = getattr(getattr(index, "gca_resource", None), "metadata", None)
    if metadata is None:
        metadata = getattr(index, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    try:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(metadata, preserving_proto_field_name=False)
    except Exception as error:
        raise RuntimeError("Vertex index metadata could not be inspected") from error


def _config_value(config: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return None


def _update_method_name(index: Any) -> str:
    value = getattr(getattr(index, "gca_resource", None), "index_update_method", None)
    if value is None:
        value = getattr(index, "index_update_method", None)
    if hasattr(value, "name"):
        return str(value.name).upper()
    if isinstance(value, str):
        return value.upper()
    if isinstance(value, int):
        try:
            from google.cloud import aiplatform_v1

            return aiplatform_v1.types.Index.IndexUpdateMethod(value).name.upper()
        except Exception as error:
            raise RuntimeError("Vertex index update method could not be decoded") from error
    raise RuntimeError("Vertex index update method is absent")


def verify_live_vertex_target(
    index: Any,
    endpoint: Any,
    *,
    expected_index_name: str,
    expected_endpoint_name: str,
    deployed_index_id: str,
    expected_dimensions: int,
    fence: Callable[[], None],
) -> None:
    """Fail closed unless the live deployment matches the manifest contract."""

    if not isinstance(expected_dimensions, int) or expected_dimensions <= 0:
        raise ValueError("expected Vertex dimensions must be positive")
    fence()
    for resource in (index, endpoint):
        sync = getattr(resource, "sync", None)
        if callable(sync):
            sync()
        fence()
    if _resource_name(index) != expected_index_name:
        raise RuntimeError("Vertex index client resolved to another resource")
    if _resource_name(endpoint) != expected_endpoint_name:
        raise RuntimeError("Vertex endpoint client resolved to another resource")

    deployed = getattr(getattr(endpoint, "gca_resource", None), "deployed_indexes", None)
    if deployed is None:
        deployed = getattr(endpoint, "deployed_indexes", None)
    matches = [
        item
        for item in list(deployed or [])
        if str(getattr(item, "id", "")) == deployed_index_id
    ]
    if len(matches) != 1:
        raise RuntimeError("deployed index id is absent or ambiguous on the live endpoint")
    mapped_index = getattr(matches[0], "index", None)
    if mapped_index != expected_index_name:
        raise RuntimeError("deployed index id maps to another Vertex index")

    metadata = _metadata_dict(index)
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Vertex index metadata has no inspectable config")
    dimensions = _config_value(config, "dimensions")
    try:
        dimensions = int(dimensions)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Vertex index dimensions are absent") from error
    if dimensions != expected_dimensions:
        raise RuntimeError("Vertex index dimensions differ from the candidate manifest")
    distance = str(
        _config_value(config, "distanceMeasureType", "distance_measure_type") or ""
    ).upper()
    if distance not in {"DOT_PRODUCT", "DOT_PRODUCT_DISTANCE"}:
        raise RuntimeError("Vertex index must use DOT_PRODUCT distance")
    if _update_method_name(index) != "STREAM_UPDATE":
        raise RuntimeError("Vertex index must use STREAM_UPDATE")
    fence()


def verify_restricted_canary(
    endpoint: Any,
    *,
    deployed_index_id: str,
    specs: Sequence[dict[str, Any]],
    generation_id: str,
    canary_datapoint_id: str,
    fence: Callable[[], None],
) -> None:
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
        Namespace,
    )

    canary = next(
        (spec for spec in specs if spec["datapoint_id"] == canary_datapoint_id),
        None,
    )
    if canary is None:
        raise RuntimeError("manifest canary is absent from the candidate generation")
    fence()
    results = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[canary["feature_vector"]],
        num_neighbors=min(10, len(specs)),
        filter=[Namespace(GENERATION_NAMESPACE, [generation_id], [])],
    )
    neighbors = results[0] if results else []
    prefix = f"{generation_id}::"
    if (
        not neighbors
        or any(not neighbor.id.startswith(prefix) for neighbor in neighbors)
    ):
        raise RuntimeError("restricted endpoint canary returned a mixed or empty generation")


def wait_for_generation_visibility(
    endpoint: Any,
    *,
    deployed_index_id: str,
    specs: Sequence[dict[str, Any]],
    generation_id: str,
    canary_datapoint_id: str,
    fence: Callable[[], None],
    timeout_seconds: float = 300.0,
    poll_seconds: float = 2.0,
    batch_size: int = 100,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll the exact deployed endpoint until every vector and restrict is visible."""
    expected = {spec["datapoint_id"]: spec for spec in specs}
    deadline = monotonic() + timeout_seconds
    while True:
        fence()
        visible: dict[str, Any] = {}
        for id_batch in _batches(list(expected), batch_size):
            fence()
            for datapoint in endpoint.read_index_datapoints(
                deployed_index_id=deployed_index_id,
                ids=list(id_batch),
            ):
                visible[datapoint.datapoint_id] = datapoint

        complete = len(visible) == len(expected)
        if complete:
            for datapoint_id, spec in expected.items():
                datapoint = visible.get(datapoint_id)
                if datapoint is None or not _vector_matches(
                    datapoint.feature_vector, spec["feature_vector"]
                ):
                    complete = False
                    break
                if _restriction_tokens(datapoint, GENERATION_NAMESPACE) != {
                    generation_id
                }:
                    complete = False
                    break
        if complete:
            fence()
            verify_restricted_canary(
                endpoint,
                deployed_index_id=deployed_index_id,
                specs=specs,
                generation_id=generation_id,
                canary_datapoint_id=canary_datapoint_id,
                fence=fence,
            )
            return
        if monotonic() >= deadline:
            missing = sorted(set(expected) - set(visible))[:5]
            raise TimeoutError(
                f"generation {generation_id} was not fully visible; missing={missing}"
            )
        sleep(poll_seconds)

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .hashing import canonical_hash


class ClaimOwnershipError(RuntimeError):
    pass


def _write_json_handle(handle, value: dict) -> None:
    json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    directory_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            _write_json_handle(handle, value)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_exclusive(path: Path, value: dict, mode: int = 0o600) -> None:
    """Create a durable JSON file exactly once without an overwrite window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            _write_json_handle(handle, value)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def update_json_claimed(
    path: Path, value: dict, attempt_owner: str, attempt_nonce: str
) -> None:
    """Update an existing claim only when its unique ownership tokens still match."""

    with path.open(encoding="utf-8") as handle:
        current = json.load(handle)
    if not isinstance(current, dict):
        raise ClaimOwnershipError("claim ledger is not a JSON object")
    if current.get("attempt_owner") != attempt_owner:
        raise ClaimOwnershipError("claim ledger owner changed")
    if current.get("attempt_nonce") != attempt_nonce:
        raise ClaimOwnershipError("claim ledger nonce changed")
    if value.get("attempt_owner") != attempt_owner:
        raise ClaimOwnershipError("updated ledger does not preserve its owner")
    if value.get("attempt_nonce") != attempt_nonce:
        raise ClaimOwnershipError("updated ledger does not preserve its nonce")
    write_json_atomic(path, value)


def build_selection_receipt(
    result: dict, config: dict, validation_artifact_sha256: str
) -> dict:
    selection = result["selection"]
    selected = selection["selected_model"]
    spec = next(spec for spec in config["models"] if spec["name"] == selected)
    baseline = selection["strongest_simple_baseline"]
    baseline_spec = next(spec for spec in config["models"] if spec["name"] == baseline)
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "validation",
        "test_accessed": False,
        "archive_sha256": result["archive_sha256"],
        "protocol_sha256": result["protocol_sha256"],
        "config_sha256": result["config_sha256"],
        "benchmark_code_sha256": result["benchmark_code_sha256"],
        "validation_split_artifact_sha256": result["split_artifact_sha256"],
        "validation_split_manifest_sha256": result["split_manifest_sha256"],
        "validation_artifact_sha256": validation_artifact_sha256,
        "selection_rule": selection["rule"],
        "primary_metric": selection["primary_metric"],
        "selected_model": selected,
        "selected_model_params": dict(spec.get("params", {})),
        "strongest_simple_baseline": baseline,
        "strongest_simple_baseline_params": dict(baseline_spec.get("params", {})),
        "validation_primary_metrics": {
            name: model["metrics"][selection["primary_metric"]]
            for name, model in result["models"].items()
        },
        "test_metrics": None,
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    return receipt


def verify_receipt(receipt: dict) -> None:
    if receipt.get("stage") != "validation" or receipt.get("test_accessed") is not False:
        raise ValueError("selection receipt is not validation-only")
    if receipt.get("test_metrics") is not None:
        raise ValueError("selection receipt unexpectedly contains test metrics")
    expected = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if expected != canonical_hash(unsigned):
        raise ValueError("selection receipt hash is invalid")


def verify_selection_against_validation(
    receipt: dict,
    validation: dict,
    validation_artifact_sha256: str,
    protocol: dict,
    config: dict,
) -> None:
    from .runner import recompute_selection

    verify_receipt(receipt)
    if receipt.get("protocol_sha256") != canonical_hash(protocol):
        raise ValueError("receipt protocol hash differs from current protocol")
    if receipt.get("config_sha256") != canonical_hash(config):
        raise ValueError("receipt config hash differs from current config")
    if receipt.get("validation_artifact_sha256") != validation_artifact_sha256:
        raise ValueError("receipt does not hash-link the validation artifact")
    for field in (
        "archive_sha256",
        "protocol_sha256",
        "config_sha256",
        "benchmark_code_sha256",
    ):
        if receipt.get(field) != validation.get(field):
            raise ValueError(f"receipt and validation differ on {field}")
    if receipt.get("validation_split_artifact_sha256") != validation.get(
        "split_artifact_sha256"
    ):
        raise ValueError("receipt and validation split artifact hashes differ")
    if receipt.get("validation_split_manifest_sha256") != validation.get(
        "split_manifest_sha256"
    ):
        raise ValueError("receipt and validation split manifest hashes differ")
    selected, strongest_simple = recompute_selection(validation, protocol, config)
    if receipt.get("selected_model") != selected:
        raise ValueError("receipt selected model is not the validation argmax")
    if receipt.get("strongest_simple_baseline") != strongest_simple:
        raise ValueError("receipt strongest simple baseline is not the validation argmax")
    primary = receipt.get("primary_metric")
    expected_metrics = {
        name: model["metrics"][primary] for name, model in validation["models"].items()
    }
    if receipt.get("validation_primary_metrics") != expected_metrics:
        raise ValueError("receipt validation metrics differ from validation artifact")
    selected_spec = next(spec for spec in config["models"] if spec["name"] == selected)
    baseline_spec = next(
        spec for spec in config["models"] if spec["name"] == strongest_simple
    )
    if receipt.get("selected_model_params") != selected_spec.get("params", {}):
        raise ValueError("receipt selected-model parameters differ from config")
    if receipt.get("strongest_simple_baseline_params") != baseline_spec.get("params", {}):
        raise ValueError("receipt baseline parameters differ from config")

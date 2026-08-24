from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from reco_benchmark.hashing import (
    benchmark_source_hash,
    canonical_hash,
    load_json,
    sha256_file,
)
from reco_benchmark.paths import (
    DEFAULT_CONFIG,
    DEFAULT_PROTOCOL,
    DEFAULT_SELECTION_RECEIPT,
    DEFAULT_TEST_ACCESS_LEDGER,
    DEFAULT_TEST_MANIFEST,
    DEFAULT_TEST_RESULT,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VALIDATION_MANIFEST,
    DEFAULT_VALIDATION_RESULT,
)
from reco_benchmark.receipts import (
    update_json_claimed,
    verify_selection_against_validation,
    write_json_exclusive,
)
from reco_benchmark.runner import add_test_comparison, run_models
from reco_benchmark.sealed_test import load_sealed_test_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process-enforced one-shot frozen-test evaluation"
    )
    parser.add_argument(
        "--protocol-commit",
        required=True,
        help="the exact checked-out commit containing frozen code and validation evidence",
    )
    return parser.parse_args()


def _run_git(arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"evidence path must be inside the repository: {path}") from error


def _committed_bytes(commit: str, path: Path) -> bytes:
    relative = _relative(path)
    committed = _run_git(["show", f"{commit}:{relative}"])
    working = path.read_bytes()
    if committed != working:
        raise ValueError(f"working {relative} differs byte-for-byte from the commit")
    return committed


def _committed_json(commit: str, path: Path) -> dict:
    value = json.loads(_committed_bytes(commit, path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"committed {_relative(path)} is not a JSON object")
    return value


def _committed_benchmark_source_hash(commit: str) -> str:
    prefix = _relative(ROOT)
    paths = _run_git(["ls-tree", "-r", "--name-only", commit, "--", prefix])
    relative_paths = paths.decode("utf-8").splitlines()
    source_paths = sorted(
        path
        for path in relative_paths
        if path.endswith(".py") or path == f"{prefix}/requirements.txt"
    )
    manifest = {}
    for repository_path in source_paths:
        value = _run_git(["show", f"{commit}:{repository_path}"])
        benchmark_path = Path(repository_path).relative_to(prefix).as_posix()
        manifest[benchmark_path] = hashlib.sha256(value).hexdigest()
    return canonical_hash(manifest)


def _resolve_exact_commit(requested: str) -> str:
    commit = _run_git(["rev-parse", "--verify", f"{requested}^{{commit}}"]).decode().strip()
    head = _run_git(["rev-parse", "HEAD"]).decode().strip()
    if commit != head:
        raise SystemExit(
            "protocol commit must equal the exact checked-out HEAD; evaluate from the frozen commit"
        )
    return commit


def _guard_first_access() -> None:
    for path in (DEFAULT_TEST_RESULT, DEFAULT_TEST_ACCESS_LEDGER):
        if path.exists():
            raise SystemExit(f"test access already recorded at {path}")
        history = _run_git(["log", "--all", "--format=%H", "--", _relative(path)])
        if history.strip():
            raise SystemExit(f"repository history already contains {_relative(path)}")


def main() -> None:
    args = parse_args()
    commit = _resolve_exact_commit(args.protocol_commit)
    _guard_first_access()

    protocol = _committed_json(commit, DEFAULT_PROTOCOL)
    config = _committed_json(commit, DEFAULT_CONFIG)
    validation = _committed_json(commit, DEFAULT_VALIDATION_RESULT)
    receipt = _committed_json(commit, DEFAULT_SELECTION_RECEIPT)
    validation_manifest = _committed_json(commit, DEFAULT_VALIDATION_MANIFEST)
    test_manifest = _committed_json(commit, DEFAULT_TEST_MANIFEST)

    validation_hash = sha256_file(DEFAULT_VALIDATION_RESULT)
    verify_selection_against_validation(
        receipt, validation, validation_hash, protocol, config
    )
    if validation.get("stage") != "validation" or validation.get("test_accessed") is not False:
        raise SystemExit("committed validation artifact is not validation-only")
    if validation.get("split_manifest_sha256") != canonical_hash(validation_manifest):
        raise SystemExit("validation artifact does not match committed split manifest")

    working_code_hash = benchmark_source_hash(ROOT)
    committed_code_hash = _committed_benchmark_source_hash(commit)
    if working_code_hash != committed_code_hash:
        raise SystemExit("working benchmark source tree differs from the frozen commit")
    if receipt.get("benchmark_code_sha256") != committed_code_hash:
        raise SystemExit("selection receipt does not match frozen benchmark source tree")

    if test_manifest.get("protocol_sha256") != canonical_hash(protocol):
        raise SystemExit("test split manifest protocol hash mismatch")
    if test_manifest.get("stage") != "test":
        raise SystemExit("test split manifest stage mismatch")
    expected_test_split_hash = test_manifest.get("artifact_sha256")
    if not isinstance(expected_test_split_hash, str) or len(expected_test_split_hash) != 64:
        raise SystemExit("test split manifest lacks a valid expected artifact hash")

    authorized_at = datetime.now(timezone.utc).isoformat()
    attempt_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
    attempt_nonce = secrets.token_hex(32)
    ledger = {
        "schema_version": 2,
        "status": "authorized_test_access",
        "authorized_at": authorized_at,
        "attempt_owner": attempt_owner,
        "attempt_nonce": attempt_nonce,
        "protocol_commit": commit,
        "selection_receipt_sha256": receipt["receipt_sha256"],
        "expected_test_split_artifact_sha256": expected_test_split_hash,
        "test_split_manifest_sha256": canonical_hash(test_manifest),
        "guard": "process-enforced repository ledger; not cryptographic access control",
    }
    # Authorization is consumed before the test artifact is opened or even hashed.
    # A mismatch, exception, or crash therefore cannot be retried as a first access.
    try:
        write_json_exclusive(DEFAULT_TEST_ACCESS_LEDGER, ledger, mode=0o600)
    except FileExistsError as error:
        raise SystemExit("another evaluator already claimed sealed-test access") from error

    try:
        test_split_hash = sha256_file(DEFAULT_TEST_SPLIT)
        ledger["actual_test_split_artifact_sha256"] = test_split_hash
        if expected_test_split_hash != test_split_hash:
            raise ValueError("test split artifact hash mismatch after authorization")

        data = load_sealed_test_data(DEFAULT_TEST_SPLIT, test_manifest, protocol)
        selected = receipt["selected_model"]
        baseline = receipt["strongest_simple_baseline"]
        result, evaluations = run_models(
            data, protocol, config, evaluated_models={selected, baseline}
        )
        add_test_comparison(result, evaluations, receipt, protocol)
        result["selection_receipt_sha256"] = receipt["receipt_sha256"]
        result["validation_artifact_sha256"] = validation_hash
        result["protocol_commit"] = commit
        result["evaluation_ordinal"] = 1
        result["test_access_authorized_at"] = authorized_at
        result["test_access_attempt_owner"] = attempt_owner
        result["test_access_attempt_nonce"] = attempt_nonce
        write_json_exclusive(DEFAULT_TEST_RESULT, result, mode=0o600)

        ledger["status"] = "completed"
        ledger["completed_at"] = datetime.now(timezone.utc).isoformat()
        ledger["test_result_sha256"] = sha256_file(DEFAULT_TEST_RESULT)
        update_json_claimed(
            DEFAULT_TEST_ACCESS_LEDGER, ledger, attempt_owner, attempt_nonce
        )
    except BaseException as error:
        ledger["status"] = "failed"
        ledger["failed_at"] = datetime.now(timezone.utc).isoformat()
        ledger["failure_type"] = type(error).__name__
        ledger["failure_message"] = str(error)
        try:
            update_json_claimed(
                DEFAULT_TEST_ACCESS_LEDGER, ledger, attempt_owner, attempt_nonce
            )
        except BaseException:
            pass
        raise
    print(f"sealed test evidence: {DEFAULT_TEST_RESULT}")
    print(f"access ledger: {DEFAULT_TEST_ACCESS_LEDGER}")


if __name__ == "__main__":
    main()

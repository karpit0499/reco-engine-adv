from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_test import _committed_benchmark_source_hash, _committed_bytes, _run_git
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
    DEFAULT_VALIDATION_MANIFEST,
    DEFAULT_VALIDATION_RESULT,
)
from reco_benchmark.receipts import verify_selection_against_validation


def verify_post_test_links(
    result: dict,
    ledger: dict,
    test_manifest: dict,
    protocol: dict,
    config: dict,
    validation: dict,
    receipt: dict,
    validation_hash: str,
    result_hash: str,
) -> None:
    if ledger.get("schema_version") != 2 or ledger.get("status") != "completed":
        raise ValueError("test access ledger is not a completed first access")
    attempt_owner = ledger.get("attempt_owner")
    attempt_nonce = ledger.get("attempt_nonce")
    if not isinstance(attempt_owner, str) or not attempt_owner:
        raise ValueError("test access ledger lacks an attempt owner")
    if (
        not isinstance(attempt_nonce, str)
        or len(attempt_nonce) != 64
        or any(character not in "0123456789abcdef" for character in attempt_nonce)
    ):
        raise ValueError("test access ledger lacks a valid attempt nonce")
    expected_split_hash = test_manifest.get("artifact_sha256")
    if ledger.get("expected_test_split_artifact_sha256") != expected_split_hash:
        raise ValueError("ledger expected test split hash differs from manifest")
    if ledger.get("actual_test_split_artifact_sha256") != expected_split_hash:
        raise ValueError("ledger actual test split hash differs from manifest")
    test_manifest_hash = canonical_hash(test_manifest)
    if ledger.get("test_split_manifest_sha256") != test_manifest_hash:
        raise ValueError("ledger test manifest hash mismatch")
    if ledger.get("selection_receipt_sha256") != receipt.get("receipt_sha256"):
        raise ValueError("ledger selection receipt link mismatch")
    if ledger.get("test_result_sha256") != result_hash:
        raise ValueError("ledger test result hash mismatch")

    if result.get("schema_version") != 2:
        raise ValueError("test result schema mismatch")
    if result.get("stage") != "test" or result.get("test_accessed") is not True:
        raise ValueError("test result is not marked as test evidence")
    if result.get("evaluation_ordinal") != 1:
        raise ValueError("test result evaluation ordinal is not one")
    if result.get("protocol_sha256") != canonical_hash(protocol):
        raise ValueError("test result protocol hash mismatch")
    if result.get("config_sha256") != canonical_hash(config):
        raise ValueError("test result config hash mismatch")
    if result.get("archive_sha256") != receipt.get("archive_sha256"):
        raise ValueError("test result archive link mismatch")
    if result.get("benchmark_code_sha256") != receipt.get("benchmark_code_sha256"):
        raise ValueError("test result benchmark source link mismatch")
    if result.get("split_artifact_sha256") != expected_split_hash:
        raise ValueError("test result split artifact hash mismatch")
    if result.get("split_manifest_sha256") != test_manifest_hash:
        raise ValueError("test result split manifest hash mismatch")
    if result.get("selection_receipt_sha256") != receipt.get("receipt_sha256"):
        raise ValueError("test result selection receipt link mismatch")
    if result.get("validation_artifact_sha256") != validation_hash:
        raise ValueError("test result validation artifact link mismatch")
    if result.get("protocol_commit") != ledger.get("protocol_commit"):
        raise ValueError("test result and ledger protocol commits differ")
    if result.get("test_access_authorized_at") != ledger.get("authorized_at"):
        raise ValueError("test result and ledger authorization timestamps differ")
    if result.get("test_access_attempt_owner") != attempt_owner:
        raise ValueError("test result and ledger attempt owners differ")
    if result.get("test_access_attempt_nonce") != attempt_nonce:
        raise ValueError("test result and ledger attempt nonces differ")

    selected = receipt["selected_model"]
    baseline = receipt["strongest_simple_baseline"]
    if set(result.get("models", {})) != {selected, baseline}:
        raise ValueError("test result does not contain exactly selected and baseline models")
    comparison = result.get("frozen_test_comparison", {})
    if comparison.get("selected_model") != selected:
        raise ValueError("test comparison selected model mismatch")
    if comparison.get("strongest_simple_baseline") != baseline:
        raise ValueError("test comparison baseline mismatch")
    primary = receipt["primary_metric"]
    primary_difference = comparison.get("paired_differences", {}).get(primary)
    if not isinstance(primary_difference, dict):
        raise ValueError("test comparison lacks the frozen primary paired difference")
    gate = comparison.get("acceptance_gate", {})
    expected_gate = primary_difference.get("ci_low", 0.0) > 0.0
    if gate.get("metric") != primary:
        raise ValueError("test acceptance gate metric mismatch")
    if gate.get("rule") != protocol["selection"]["test_acceptance_gate"]:
        raise ValueError("test acceptance gate rule mismatch")
    if gate.get("passed") is not expected_gate:
        raise ValueError("test acceptance gate result does not match its paired interval")
    if validation.get("test_accessed") is not False:
        raise ValueError("validation artifact was not validation-only")


def _verify_frozen_protocol_commit(commit: str, receipt: dict) -> None:
    resolved = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"]).decode().strip()
    head = _run_git(["rev-parse", "HEAD"]).decode().strip()
    try:
        _run_git(["merge-base", "--is-ancestor", resolved, head])
    except subprocess.CalledProcessError as error:
        raise ValueError("recorded protocol commit is not an ancestor of HEAD") from error
    for path in (
        DEFAULT_PROTOCOL,
        DEFAULT_CONFIG,
        DEFAULT_VALIDATION_RESULT,
        DEFAULT_SELECTION_RECEIPT,
        DEFAULT_VALIDATION_MANIFEST,
        DEFAULT_TEST_MANIFEST,
    ):
        _committed_bytes(resolved, path)
    if _committed_benchmark_source_hash(resolved) != receipt.get(
        "benchmark_code_sha256"
    ):
        raise ValueError("recorded protocol commit benchmark source hash mismatch")


def _verify_post_evidence_committed() -> None:
    head = _run_git(["rev-parse", "HEAD"]).decode().strip()
    _committed_bytes(head, DEFAULT_TEST_RESULT)
    _committed_bytes(head, DEFAULT_TEST_ACCESS_LEDGER)


def main() -> None:
    protocol = load_json(DEFAULT_PROTOCOL)
    config = load_json(DEFAULT_CONFIG)
    validation = load_json(DEFAULT_VALIDATION_RESULT)
    receipt = load_json(DEFAULT_SELECTION_RECEIPT)
    validation_manifest = load_json(DEFAULT_VALIDATION_MANIFEST)
    validation_hash = sha256_file(DEFAULT_VALIDATION_RESULT)
    verify_selection_against_validation(
        receipt, validation, validation_hash, protocol, config
    )
    if validation.get("benchmark_code_sha256") != benchmark_source_hash(ROOT):
        raise SystemExit("validation evidence benchmark source hash mismatch")
    if validation.get("split_manifest_sha256") != canonical_hash(validation_manifest):
        raise SystemExit("validation evidence split manifest hash mismatch")

    result_exists = DEFAULT_TEST_RESULT.exists()
    ledger_exists = DEFAULT_TEST_ACCESS_LEDGER.exists()
    if result_exists != ledger_exists:
        raise SystemExit("test result and access ledger must either both exist or both be absent")
    if not result_exists:
        print(f"validation evidence verified: {validation_hash}")
        print(f"selected model: {receipt['selected_model']}")
        print(f"strongest simple baseline: {receipt['strongest_simple_baseline']}")
        print("sealed test accessed: false")
        return

    result = load_json(DEFAULT_TEST_RESULT)
    ledger = load_json(DEFAULT_TEST_ACCESS_LEDGER)
    test_manifest = load_json(DEFAULT_TEST_MANIFEST)
    result_hash = sha256_file(DEFAULT_TEST_RESULT)
    verify_post_test_links(
        result,
        ledger,
        test_manifest,
        protocol,
        config,
        validation,
        receipt,
        validation_hash,
        result_hash,
    )
    _verify_frozen_protocol_commit(ledger["protocol_commit"], receipt)
    _verify_post_evidence_committed()
    gate = result["frozen_test_comparison"]["acceptance_gate"]
    print(f"validation evidence verified: {validation_hash}")
    print(f"test evidence verified: {result_hash}")
    print(f"test acceptance gate passed: {str(gate['passed']).lower()}")


if __name__ == "__main__":
    main()

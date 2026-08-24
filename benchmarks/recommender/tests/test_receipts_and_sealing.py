from __future__ import annotations

import inspect
import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_test
import verify_validation_evidence
from reco_benchmark import validation_data
from reco_benchmark.hashing import benchmark_source_hash, canonical_hash
from reco_benchmark.receipts import (
    ClaimOwnershipError,
    build_selection_receipt,
    update_json_claimed,
    verify_receipt,
    verify_selection_against_validation,
    write_json_exclusive,
)


def _validation_evidence() -> tuple[dict, dict, dict]:
    config = {
        "models": [
            {"name": "popularity", "params": {}},
            {"name": "item_knn", "params": {"neighbors": 10}},
            {"name": "als", "params": {"factors": 4}},
        ]
    }
    protocol = {
        "evaluation": {
            "primary_metric": "ndcg_at_10",
            "simple_baselines": ["popularity", "item_knn"],
        }
    }
    estimates = {"popularity": 0.1, "item_knn": 0.2, "als": 0.3}
    validation = {
        "stage": "validation",
        "test_accessed": False,
        "archive_sha256": "archive",
        "protocol_sha256": canonical_hash(protocol),
        "config_sha256": canonical_hash(config),
        "benchmark_code_sha256": "code",
        "split_artifact_sha256": "split",
        "split_manifest_sha256": "manifest",
        "models": {
            name: {
                "metrics": {
                    "ndcg_at_10": {
                        "estimate": estimate,
                        "ci_low": estimate - 0.01,
                        "ci_high": estimate + 0.01,
                    }
                }
            }
            for name, estimate in estimates.items()
        },
        "selection": {
            "rule": "max",
            "primary_metric": "ndcg_at_10",
            "selected_model": "als",
            "strongest_simple_baseline": "item_knn",
        },
    }
    return validation, protocol, config


def test_validation_loader_has_no_raw_or_test_data_dependency() -> None:
    source = inspect.getsource(validation_data)
    assert "build_global_split" not in source
    assert "verify_archive" not in source
    assert "sealed_test" not in source
    assert "load_prepared_split" in source


def test_receipt_hash_links_validation_and_recomputes_argmax() -> None:
    validation, protocol, config = _validation_evidence()
    receipt = build_selection_receipt(validation, config, "validation-hash")
    verify_selection_against_validation(
        receipt, validation, "validation-hash", protocol, config
    )
    receipt["selected_model"] = "popularity"
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_hash(unsigned)
    with pytest.raises(ValueError, match="not the validation argmax"):
        verify_selection_against_validation(
            receipt, validation, "validation-hash", protocol, config
        )


def test_receipt_hash_detects_tampering() -> None:
    validation, _, config = _validation_evidence()
    receipt = build_selection_receipt(validation, config, "validation-hash")
    verify_receipt(receipt)
    receipt["selected_model"] = "popularity"
    with pytest.raises(ValueError, match="hash is invalid"):
        verify_receipt(receipt)


def test_benchmark_source_hash_covers_code_and_requirements(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy==1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("first\n", encoding="utf-8")
    initial = benchmark_source_hash(tmp_path)
    (tmp_path / "README.md").write_text("second\n", encoding="utf-8")
    assert benchmark_source_hash(tmp_path) == initial
    (tmp_path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert benchmark_source_hash(tmp_path) != initial


def test_access_guard_refuses_existing_ledger(monkeypatch, tmp_path: Path) -> None:
    result = tmp_path / "test.json"
    ledger = tmp_path / "test_access_ledger.json"
    monkeypatch.setattr(evaluate_test, "DEFAULT_TEST_RESULT", result)
    monkeypatch.setattr(evaluate_test, "DEFAULT_TEST_ACCESS_LEDGER", ledger)
    monkeypatch.setattr(evaluate_test, "_run_git", lambda arguments: b"")
    monkeypatch.setattr(evaluate_test, "_relative", lambda path: path.name)
    evaluate_test._guard_first_access()
    ledger.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="already recorded"):
        evaluate_test._guard_first_access()


def test_test_attempt_is_consumed_before_artifact_hash(monkeypatch, tmp_path: Path) -> None:
    paths = {
        "protocol": tmp_path / "protocol.json",
        "config": tmp_path / "config.json",
        "validation": tmp_path / "validation.json",
        "receipt": tmp_path / "receipt.json",
        "validation_manifest": tmp_path / "validation_manifest.json",
        "test_manifest": tmp_path / "test_manifest.json",
        "test_split": tmp_path / "test.npz",
        "test_result": tmp_path / "test.json",
        "ledger": tmp_path / "test_access_ledger.json",
    }
    protocol = {"selection": {"test_acceptance_gate": "lower bound > 0"}}
    config = {}
    validation_manifest = {"stage": "validation"}
    validation = {
        "stage": "validation",
        "test_accessed": False,
        "split_manifest_sha256": canonical_hash(validation_manifest),
    }
    receipt = {"receipt_sha256": "receipt", "benchmark_code_sha256": "code"}
    test_manifest = {
        "stage": "test",
        "protocol_sha256": canonical_hash(protocol),
        "artifact_sha256": "a" * 64,
    }
    values = {
        paths["protocol"]: protocol,
        paths["config"]: config,
        paths["validation"]: validation,
        paths["receipt"]: receipt,
        paths["validation_manifest"]: validation_manifest,
        paths["test_manifest"]: test_manifest,
    }
    for name, constant in (
        ("DEFAULT_PROTOCOL", "protocol"),
        ("DEFAULT_CONFIG", "config"),
        ("DEFAULT_VALIDATION_RESULT", "validation"),
        ("DEFAULT_SELECTION_RECEIPT", "receipt"),
        ("DEFAULT_VALIDATION_MANIFEST", "validation_manifest"),
        ("DEFAULT_TEST_MANIFEST", "test_manifest"),
        ("DEFAULT_TEST_SPLIT", "test_split"),
        ("DEFAULT_TEST_RESULT", "test_result"),
        ("DEFAULT_TEST_ACCESS_LEDGER", "ledger"),
    ):
        monkeypatch.setattr(evaluate_test, name, paths[constant])
    monkeypatch.setattr(
        evaluate_test, "parse_args", lambda: SimpleNamespace(protocol_commit="frozen")
    )
    monkeypatch.setattr(evaluate_test, "_resolve_exact_commit", lambda value: "commit")
    monkeypatch.setattr(evaluate_test, "_guard_first_access", lambda: None)
    monkeypatch.setattr(evaluate_test, "_committed_json", lambda commit, path: values[path])
    monkeypatch.setattr(
        evaluate_test, "verify_selection_against_validation", lambda *args: None
    )
    monkeypatch.setattr(evaluate_test, "benchmark_source_hash", lambda root: "code")
    monkeypatch.setattr(
        evaluate_test, "_committed_benchmark_source_hash", lambda commit: "code"
    )

    def fake_hash(path: Path) -> str:
        if path == paths["validation"]:
            return "validation-hash"
        if path == paths["test_split"]:
            assert paths["ledger"].exists()
            authorized = json.loads(paths["ledger"].read_text(encoding="utf-8"))
            assert authorized["status"] == "authorized_test_access"
            return "b" * 64
        raise AssertionError(f"unexpected hash read: {path}")

    monkeypatch.setattr(evaluate_test, "sha256_file", fake_hash)
    monkeypatch.setattr(
        evaluate_test,
        "load_sealed_test_data",
        lambda *args: pytest.fail("test artifact was loaded after a hash mismatch"),
    )
    with pytest.raises(ValueError, match="after authorization"):
        evaluate_test.main()
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    assert ledger["status"] == "failed"
    assert ledger["expected_test_split_artifact_sha256"] == "a" * 64
    assert ledger["actual_test_split_artifact_sha256"] == "b" * 64
    assert not paths["test_result"].exists()


def test_post_test_verifier_checks_links_ordinal_and_gate() -> None:
    protocol = {"selection": {"test_acceptance_gate": "lower bound > 0"}}
    config = {"candidate_policy": "fixed"}
    validation = {"test_accessed": False}
    receipt = {
        "receipt_sha256": "receipt",
        "archive_sha256": "archive",
        "benchmark_code_sha256": "code",
        "selected_model": "als",
        "strongest_simple_baseline": "popularity",
        "primary_metric": "ndcg_at_10",
    }
    test_manifest = {"artifact_sha256": "split"}
    manifest_hash = canonical_hash(test_manifest)
    result = {
        "schema_version": 2,
        "stage": "test",
        "test_accessed": True,
        "evaluation_ordinal": 1,
        "protocol_sha256": canonical_hash(protocol),
        "config_sha256": canonical_hash(config),
        "archive_sha256": "archive",
        "benchmark_code_sha256": "code",
        "split_artifact_sha256": "split",
        "split_manifest_sha256": manifest_hash,
        "selection_receipt_sha256": "receipt",
        "validation_artifact_sha256": "validation-hash",
        "protocol_commit": "commit",
        "test_access_authorized_at": "time",
        "models": {"als": {}, "popularity": {}},
        "frozen_test_comparison": {
            "selected_model": "als",
            "strongest_simple_baseline": "popularity",
            "paired_differences": {"ndcg_at_10": {"ci_low": 0.01}},
            "acceptance_gate": {
                "metric": "ndcg_at_10",
                "rule": "lower bound > 0",
                "passed": True,
            },
        },
    }
    attempt_nonce = "a" * 64
    ledger = {
        "schema_version": 2,
        "status": "completed",
        "attempt_owner": "owner",
        "attempt_nonce": attempt_nonce,
        "expected_test_split_artifact_sha256": "split",
        "actual_test_split_artifact_sha256": "split",
        "test_split_manifest_sha256": manifest_hash,
        "selection_receipt_sha256": "receipt",
        "test_result_sha256": "result-hash",
        "protocol_commit": "commit",
        "authorized_at": "time",
    }
    result["test_access_attempt_owner"] = "owner"
    result["test_access_attempt_nonce"] = attempt_nonce
    verify_validation_evidence.verify_post_test_links(
        result,
        ledger,
        test_manifest,
        protocol,
        config,
        validation,
        receipt,
        "validation-hash",
        "result-hash",
    )
    result["frozen_test_comparison"]["acceptance_gate"]["passed"] = False
    with pytest.raises(ValueError, match="does not match"):
        verify_validation_evidence.verify_post_test_links(
            result,
            ledger,
            test_manifest,
            protocol,
            config,
            validation,
            receipt,
            "validation-hash",
            "result-hash",
        )


def test_exclusive_ledger_claim_has_one_concurrent_winner(tmp_path: Path) -> None:
    ledger_path = tmp_path / "test_access_ledger.json"
    workers = 8
    barrier = threading.Barrier(workers)

    def claim(index: int) -> tuple[int, bool]:
        value = {
            "attempt_owner": f"owner-{index}",
            "attempt_nonce": f"{index:064x}",
        }
        barrier.wait()
        try:
            write_json_exclusive(ledger_path, value, mode=0o600)
            return index, True
        except FileExistsError:
            return index, False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(claim, range(workers)))
    winners = [index for index, won in outcomes if won]
    assert len(winners) == 1
    winner = winners[0]
    claimed = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert claimed["attempt_owner"] == f"owner-{winner}"
    assert claimed["attempt_nonce"] == f"{winner:064x}"
    assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o600

    original = ledger_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_exclusive(
            ledger_path,
            {"attempt_owner": "late", "attempt_nonce": "f" * 64},
            mode=0o600,
        )
    assert ledger_path.read_bytes() == original


def test_claim_updates_require_and_preserve_owner_nonce(tmp_path: Path) -> None:
    ledger_path = tmp_path / "test_access_ledger.json"
    owner = "unique-owner"
    nonce = "a" * 64
    initial = {
        "status": "authorized_test_access",
        "attempt_owner": owner,
        "attempt_nonce": nonce,
    }
    write_json_exclusive(ledger_path, initial, mode=0o600)
    changed = dict(initial, status="completed")
    with pytest.raises(ClaimOwnershipError, match="nonce changed"):
        update_json_claimed(ledger_path, changed, owner, "b" * 64)
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == initial
    update_json_claimed(ledger_path, changed, owner, nonce)
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == changed
    assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o600

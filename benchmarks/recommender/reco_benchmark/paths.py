from __future__ import annotations

from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = BENCHMARK_ROOT / "protocol.json"
DEFAULT_CONFIG = BENCHMARK_ROOT / "configs" / "validation.json"
DEFAULT_ARCHIVE = BENCHMARK_ROOT / "data" / "ml-1m.zip"
DEFAULT_VALIDATION_SPLIT = BENCHMARK_ROOT / "data" / "prepared" / "validation.npz"
DEFAULT_TEST_SPLIT = BENCHMARK_ROOT / "data" / "prepared" / "test.npz"
DEFAULT_VALIDATION_MANIFEST = BENCHMARK_ROOT / "splits" / "validation_manifest.json"
DEFAULT_TEST_MANIFEST = BENCHMARK_ROOT / "splits" / "test_manifest.json"
DEFAULT_VALIDATION_RESULT = BENCHMARK_ROOT / "results" / "validation.json"
DEFAULT_SELECTION_RECEIPT = BENCHMARK_ROOT / "results" / "selection_receipt.json"
DEFAULT_TEST_RESULT = BENCHMARK_ROOT / "results" / "test.json"
DEFAULT_TEST_ACCESS_LEDGER = BENCHMARK_ROOT / "results" / "test_access_ledger.json"

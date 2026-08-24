from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reco_benchmark.hashing import load_json, sha256_file
from reco_benchmark.paths import (
    DEFAULT_CONFIG,
    DEFAULT_PROTOCOL,
    DEFAULT_SELECTION_RECEIPT,
    DEFAULT_VALIDATION_MANIFEST,
    DEFAULT_VALIDATION_RESULT,
    DEFAULT_VALIDATION_SPLIT,
)
from reco_benchmark.receipts import build_selection_receipt, write_json_atomic
from reco_benchmark.runner import add_validation_selection, run_models
from reco_benchmark.validation_data import load_validation_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and select recommenders using the validation-only artifact"
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATION_RESULT)
    parser.add_argument(
        "--selection-output", type=Path, default=DEFAULT_SELECTION_RECEIPT
    )
    parser.add_argument(
        "--replace-results",
        action="store_true",
        help="replace validation-only evidence before protocol freeze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing = [
        str(path) for path in (args.output, args.selection_output) if path.exists()
    ]
    if existing and not args.replace_results:
        raise SystemExit(
            "refusing to overwrite validation evidence; pass --replace-results: "
            + ", ".join(existing)
        )
    protocol = load_json(args.protocol)
    config = load_json(args.config)
    manifest = load_json(args.split_manifest)
    data = load_validation_data(args.split, manifest, protocol)
    result, evaluations = run_models(data, protocol, config)
    add_validation_selection(result, evaluations, protocol, config)
    write_json_atomic(args.output, result)
    validation_hash = sha256_file(args.output)
    receipt = build_selection_receipt(result, config, validation_hash)
    write_json_atomic(args.selection_output, receipt)
    print(f"validation evidence: {args.output}")
    print(f"validation SHA-256: {validation_hash}")
    print(f"selection receipt: {args.selection_output}")
    print("sealed test accessed: false")


if __name__ == "__main__":
    main()

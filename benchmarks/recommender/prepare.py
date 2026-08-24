from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reco_benchmark.dataset import (
    build_global_split,
    build_split_manifest,
    download_archive,
    verify_archive,
    write_split_artifact,
)
from reco_benchmark.hashing import canonical_hash, load_json
from reco_benchmark.paths import (
    DEFAULT_ARCHIVE,
    DEFAULT_PROTOCOL,
    DEFAULT_TEST_MANIFEST,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VALIDATION_MANIFEST,
    DEFAULT_VALIDATION_SPLIT,
)
from reco_benchmark.receipts import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MovieLens and prepare physically separate temporal splits"
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--validation-split", type=Path, default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument("--test-split", type=Path, default=DEFAULT_TEST_SPLIT)
    parser.add_argument(
        "--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="fail instead of downloading when the raw archive is absent",
    )
    parser.add_argument(
        "--replace-prepared",
        action="store_true",
        help="replace existing prepared artifacts and manifests",
    )
    return parser.parse_args()


def _guard_outputs(args: argparse.Namespace) -> None:
    outputs = (
        args.validation_split,
        args.test_split,
        args.validation_manifest,
        args.test_manifest,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.replace_prepared:
        raise SystemExit(
            "refusing to replace prepared evidence; pass --replace-prepared: "
            + ", ".join(existing)
        )


def main() -> None:
    args = parse_args()
    _guard_outputs(args)
    protocol = load_json(args.protocol)
    if args.archive.exists() or args.verify_only:
        archive_sha = verify_archive(args.archive, protocol)
    else:
        archive_sha = download_archive(args.archive, protocol)

    manifests = {}
    summaries = {}
    for stage, artifact_path, manifest_path in (
        ("validation", args.validation_split, args.validation_manifest),
        ("test", args.test_split, args.test_manifest),
    ):
        data = build_global_split(args.archive, protocol, stage=stage)
        artifact_hash = write_split_artifact(artifact_path, data)
        manifest = build_split_manifest(data, artifact_path, artifact_hash)
        write_json_atomic(manifest_path, manifest)
        manifests[stage] = {
            "artifact": str(artifact_path.resolve()),
            "artifact_sha256": artifact_hash,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": canonical_hash(manifest),
        }
        summaries[stage] = data.summary()

    print(
        json.dumps(
            {
                "archive_sha256": archive_sha,
                "protocol_sha256": canonical_hash(protocol),
                "raw_data_redistributed": False,
                "splits": manifests,
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def benchmark_source_manifest(benchmark_root: Path) -> dict[str, str]:
    """Hash every executable benchmark source plus its pinned requirements."""

    paths = sorted(
        path
        for path in benchmark_root.rglob("*")
        if path.is_file()
        and (path.suffix == ".py" or path.relative_to(benchmark_root).as_posix() == "requirements.txt")
    )
    return {
        path.relative_to(benchmark_root).as_posix(): sha256_file(path) for path in paths
    }


def benchmark_source_hash(benchmark_root: Path) -> str:
    return canonical_hash(benchmark_source_manifest(benchmark_root))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value

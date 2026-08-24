from __future__ import annotations

import io
import os
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from scipy import sparse

from .hashing import canonical_hash, sha256_file


@dataclass(frozen=True)
class RatingInteraction:
    user_id: int
    movie_id: int
    rating: int
    timestamp: int


@dataclass(frozen=True)
class Catalog:
    movie_ids: np.ndarray
    titles: tuple[str, ...]
    genres: tuple[str, ...]
    movie_to_index: dict[int, int]

    @property
    def size(self) -> int:
        return int(self.movie_ids.size)


@dataclass(frozen=True)
class EvaluationData:
    stage: Literal["validation", "test"]
    catalog: Catalog
    training_user_ids: np.ndarray
    evaluation_user_indices: np.ndarray
    fit_matrix: sparse.csr_matrix
    seen_matrix: sparse.csr_matrix
    targets: np.ndarray
    target_timestamps: np.ndarray
    cold_users: np.ndarray
    archive_sha256: str
    protocol_hash: str
    split_artifact_sha256: str
    split_manifest_sha256: str
    protocol: dict

    @property
    def evaluation_user_ids(self) -> np.ndarray:
        return self.training_user_ids[self.evaluation_user_indices]

    def summary(self) -> dict:
        item_counts = np.asarray(self.fit_matrix.sum(axis=0)).ravel()
        warm_targets = item_counts[self.targets] > 0
        return {
            "stage": self.stage,
            "training_users": int(self.training_user_ids.size),
            "evaluation_users": int(self.targets.size),
            "warm_users": int(np.count_nonzero(~self.cold_users)),
            "cold_users": int(np.count_nonzero(self.cold_users)),
            "catalog_items": self.catalog.size,
            "fit_positive_interactions": int(self.fit_matrix.nnz),
            "seen_rating_events": int(self.seen_matrix.nnz),
            "warm_catalog_items": int(np.count_nonzero(item_counts)),
            "cold_catalog_items": int(np.count_nonzero(item_counts == 0)),
            "warm_targets": int(np.count_nonzero(warm_targets)),
            "cold_targets": int(np.count_nonzero(~warm_targets)),
            "target_timestamp_min": int(self.target_timestamps.min()),
            "target_timestamp_max": int(self.target_timestamps.max()),
        }


def verify_archive(archive: Path, protocol: dict) -> str:
    if not archive.is_file():
        raise FileNotFoundError(
            f"MovieLens archive not found at {archive}. Run prepare.py or pass --archive."
        )
    actual = sha256_file(archive)
    expected = protocol["dataset"]["archive_sha256"]
    if actual != expected:
        raise ValueError(
            f"MovieLens archive SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def download_archive(destination: Path, protocol: dict) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        protocol["dataset"]["source_url"],
        headers={"User-Agent": "reco-engine-adv-evidence/2.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = sha256_file(temporary)
        expected = protocol["dataset"]["archive_sha256"]
        if actual != expected:
            raise ValueError(
                f"downloaded archive SHA-256 mismatch: expected {expected}, found {actual}"
            )
        temporary.replace(destination)
        return actual
    finally:
        if temporary.exists():
            temporary.unlink()


def _text_member(archive: zipfile.ZipFile, member: str) -> io.TextIOWrapper:
    try:
        raw = archive.open(member, "r")
    except KeyError as error:
        raise ValueError(f"archive is missing {member}") from error
    return io.TextIOWrapper(raw, encoding="latin-1", newline="")


def load_catalog(archive_path: Path, protocol: dict) -> Catalog:
    rows: list[tuple[int, str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        with _text_member(archive, protocol["dataset"]["movies_member"]) as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.rstrip("\r\n").split("::")
                if len(parts) != 3:
                    raise ValueError(f"invalid movies.dat row {line_number}")
                rows.append((int(parts[0]), parts[1], parts[2]))
    rows.sort(key=lambda row: row[0])
    movie_ids = np.asarray([row[0] for row in rows], dtype=np.int32)
    if movie_ids.size != np.unique(movie_ids).size:
        raise ValueError("movies.dat contains duplicate movie IDs")
    expected = int(protocol["dataset"]["expected_catalog_items"])
    if movie_ids.size != expected:
        raise ValueError(f"expected {expected} catalog movies, found {movie_ids.size}")
    return Catalog(
        movie_ids=movie_ids,
        titles=tuple(row[1] for row in rows),
        genres=tuple(row[2] for row in rows),
        movie_to_index={int(movie_id): index for index, movie_id in enumerate(movie_ids)},
    )


def iter_rating_interactions(
    archive_path: Path, protocol: dict
) -> Iterator[RatingInteraction]:
    member = protocol["dataset"]["ratings_member"]
    with zipfile.ZipFile(archive_path) as archive:
        with _text_member(archive, member) as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.rstrip("\r\n").split("::")
                if len(parts) != 4:
                    raise ValueError(f"invalid ratings.dat row {line_number}")
                user_id, movie_id, rating, timestamp = map(int, parts)
                yield RatingInteraction(user_id, movie_id, rating, timestamp)


def _stage_bounds(
    protocol: dict, stage: Literal["validation", "test"]
) -> tuple[int, int | None]:
    temporal = protocol["temporal_split"]
    if stage == "validation":
        return int(temporal["train_cutoff_timestamp"]), int(
            temporal["test_cutoff_timestamp"]
        )
    if stage == "test":
        return int(temporal["test_cutoff_timestamp"]), None
    raise ValueError(f"unsupported stage: {stage}")


def _csr(
    rows: list[int], columns: list[int], shape: tuple[int, int], dtype: np.dtype
) -> sparse.csr_matrix:
    values = np.ones(len(rows), dtype=dtype)
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=shape, dtype=dtype)
    matrix.sum_duplicates()
    if matrix.data.size and not np.all(matrix.data == 1):
        raise ValueError("duplicate user/movie interactions remain after validation")
    matrix.sort_indices()
    return matrix


def build_global_split(
    archive_path: Path,
    protocol: dict,
    stage: Literal["validation", "test"],
) -> EvaluationData:
    """Build a globally ordered split during protocol preparation only.

    Equal timestamps are never divided across the fit and target windows. For each
    eligible user, the target is selected from that user's first timestamp block in
    the stage window. No later-window interaction is serialized into the split.
    """

    archive_sha = verify_archive(archive_path, protocol)
    catalog = load_catalog(archive_path, protocol)
    interactions = list(iter_rating_interactions(archive_path, protocol))
    if not interactions:
        raise ValueError("ratings.dat contains no interactions")

    seen_pairs: set[tuple[int, int]] = set()
    by_user: dict[int, list[RatingInteraction]] = defaultdict(list)
    for interaction in interactions:
        if interaction.movie_id not in catalog.movie_to_index:
            raise ValueError(f"rating references unknown movie {interaction.movie_id}")
        key = (interaction.user_id, interaction.movie_id)
        if key in seen_pairs:
            raise ValueError(
                f"duplicate user/movie interaction: user={key[0]}, movie={key[1]}"
            )
        seen_pairs.add(key)
        by_user[interaction.user_id].append(interaction)

    training_user_ids = np.asarray(sorted(by_user), dtype=np.int32)
    user_to_index = {
        int(user_id): index for index, user_id in enumerate(training_user_ids)
    }
    fit_cutoff, target_end = _stage_bounds(protocol, stage)
    minimum_rating = int(protocol["relevance"]["minimum_rating"])
    warm_minimum = int(protocol["cohorts"]["warm_minimum_positive_interactions"])
    cold_maximum = int(protocol["cohorts"]["cold_maximum_observed_interactions"])

    fit_rows: list[int] = []
    fit_columns: list[int] = []
    seen_rows: list[int] = []
    seen_columns: list[int] = []
    evaluation_user_indices: list[int] = []
    targets: list[int] = []
    target_timestamps: list[int] = []
    cold_users: list[bool] = []

    for user_id in training_user_ids:
        user_index = user_to_index[int(user_id)]
        sequence = sorted(
            by_user[int(user_id)], key=lambda value: (value.timestamp, value.movie_id)
        )
        before = [value for value in sequence if value.timestamp < fit_cutoff]
        window = [
            value
            for value in sequence
            if value.timestamp >= fit_cutoff
            and (target_end is None or value.timestamp < target_end)
        ]

        for interaction in before:
            item_index = catalog.movie_to_index[interaction.movie_id]
            seen_rows.append(user_index)
            seen_columns.append(item_index)
            if interaction.rating >= minimum_rating:
                fit_rows.append(user_index)
                fit_columns.append(item_index)

        if not window:
            continue
        first_timestamp = min(value.timestamp for value in window)
        first_block = [value for value in window if value.timestamp == first_timestamp]
        relevant = [value for value in first_block if value.rating >= minimum_rating]
        if not relevant:
            continue

        positive_count = sum(value.rating >= minimum_rating for value in before)
        observed_count = len(before)
        is_warm = positive_count >= warm_minimum
        is_cold = observed_count <= cold_maximum
        if not (is_warm or is_cold):
            continue

        target = min(relevant, key=lambda value: value.movie_id)
        evaluation_user_indices.append(user_index)
        targets.append(catalog.movie_to_index[target.movie_id])
        target_timestamps.append(target.timestamp)
        cold_users.append(is_cold)

    shape = (training_user_ids.size, catalog.size)
    fit_matrix = _csr(fit_rows, fit_columns, shape, np.float32)
    seen_matrix = _csr(seen_rows, seen_columns, shape, np.int8)
    evaluation_indices_array = np.asarray(evaluation_user_indices, dtype=np.int32)
    target_array = np.asarray(targets, dtype=np.int32)
    if not target_array.size:
        raise ValueError(f"{stage} split has no eligible targets")
    for user_index, target in zip(evaluation_indices_array, target_array, strict=True):
        if seen_matrix[int(user_index), int(target)] != 0:
            raise ValueError("evaluation target appears in the all-observed seen set")

    return EvaluationData(
        stage=stage,
        catalog=catalog,
        training_user_ids=training_user_ids,
        evaluation_user_indices=evaluation_indices_array,
        fit_matrix=fit_matrix,
        seen_matrix=seen_matrix,
        targets=target_array,
        target_timestamps=np.asarray(target_timestamps, dtype=np.int64),
        cold_users=np.asarray(cold_users, dtype=bool),
        archive_sha256=archive_sha,
        protocol_hash=canonical_hash(protocol),
        split_artifact_sha256="",
        split_manifest_sha256="",
        protocol=protocol,
    )


def _matrix_arrays(prefix: str, matrix: sparse.csr_matrix) -> dict[str, np.ndarray]:
    matrix = matrix.tocsr()
    return {
        f"{prefix}_data": matrix.data,
        f"{prefix}_indices": matrix.indices.astype(np.int32, copy=False),
        f"{prefix}_indptr": matrix.indptr.astype(np.int64, copy=False),
        f"{prefix}_shape": np.asarray(matrix.shape, dtype=np.int64),
    }


def write_split_artifact(path: Path, data: EvaluationData) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "artifact_schema_version": np.asarray([2], dtype=np.int16),
        "stage": np.asarray([data.stage]),
        "archive_sha256": np.asarray([data.archive_sha256]),
        "protocol_sha256": np.asarray([data.protocol_hash]),
        "catalog_movie_ids": data.catalog.movie_ids,
        "catalog_titles": np.asarray(data.catalog.titles),
        "catalog_genres": np.asarray(data.catalog.genres),
        "training_user_ids": data.training_user_ids,
        "evaluation_user_indices": data.evaluation_user_indices,
        "targets": data.targets,
        "target_timestamps": data.target_timestamps,
        "cold_users": data.cold_users,
    }
    arrays.update(_matrix_arrays("fit", data.fit_matrix))
    arrays.update(_matrix_arrays("seen", data.seen_matrix))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def build_split_manifest(
    data: EvaluationData, artifact_path: Path, artifact_sha256: str
) -> dict:
    fit_cutoff, target_end = _stage_bounds(data.protocol, data.stage)
    return {
        "schema_version": 2,
        "stage": data.stage,
        "artifact_path": data.protocol["prepared_splits"][f"{data.stage}_artifact"],
        "artifact_sha256": artifact_sha256,
        "archive_sha256": data.archive_sha256,
        "protocol_sha256": data.protocol_hash,
        "fit_timestamp_exclusive_upper_bound": fit_cutoff,
        "target_timestamp_inclusive_lower_bound": fit_cutoff,
        "target_timestamp_exclusive_upper_bound": target_end,
        "equal_timestamp_blocks_split": False,
        "contains_other_stage_targets": False,
        "raw_archive_redistributed": False,
        "artifact_filename": artifact_path.name,
        "dataset": data.summary(),
    }


def _load_csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sparse.csr_matrix:
    shape = tuple(int(value) for value in archive[f"{prefix}_shape"])
    matrix = sparse.csr_matrix(
        (
            archive[f"{prefix}_data"],
            archive[f"{prefix}_indices"],
            archive[f"{prefix}_indptr"],
        ),
        shape=shape,
    )
    matrix.sort_indices()
    return matrix


def load_prepared_split(
    artifact_path: Path,
    manifest: dict,
    protocol: dict,
    expected_stage: Literal["validation", "test"],
) -> EvaluationData:
    if manifest.get("stage") != expected_stage:
        raise ValueError("split manifest stage mismatch")
    if manifest.get("contains_other_stage_targets") is not False:
        raise ValueError("split manifest does not guarantee stage isolation")
    protocol_hash = canonical_hash(protocol)
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("split manifest protocol hash mismatch")
    if manifest.get("archive_sha256") != protocol["dataset"]["archive_sha256"]:
        raise ValueError("split manifest archive hash mismatch")
    expected_artifact_path = protocol["prepared_splits"][f"{expected_stage}_artifact"]
    if manifest.get("artifact_path") != expected_artifact_path:
        raise ValueError("split manifest artifact path mismatch")
    if manifest.get("equal_timestamp_blocks_split") is not False:
        raise ValueError("split manifest permits an equal-timestamp boundary split")
    actual_artifact_hash = sha256_file(artifact_path)
    if manifest.get("artifact_sha256") != actual_artifact_hash:
        raise ValueError("prepared split artifact hash mismatch")

    with np.load(artifact_path, allow_pickle=False) as archive:
        schema_version = int(archive["artifact_schema_version"][0])
        stage = str(archive["stage"][0])
        archive_sha = str(archive["archive_sha256"][0])
        embedded_protocol_hash = str(archive["protocol_sha256"][0])
        if schema_version != 2 or stage != expected_stage:
            raise ValueError("prepared split metadata mismatch")
        if embedded_protocol_hash != protocol_hash:
            raise ValueError("prepared split protocol hash mismatch")
        if archive_sha != protocol["dataset"]["archive_sha256"]:
            raise ValueError("prepared split archive hash mismatch")

        movie_ids = archive["catalog_movie_ids"].astype(np.int32, copy=True)
        catalog = Catalog(
            movie_ids=movie_ids,
            titles=tuple(str(value) for value in archive["catalog_titles"]),
            genres=tuple(str(value) for value in archive["catalog_genres"]),
            movie_to_index={
                int(movie_id): index for index, movie_id in enumerate(movie_ids)
            },
        )
        data = EvaluationData(
            stage=expected_stage,
            catalog=catalog,
            training_user_ids=archive["training_user_ids"].astype(np.int32, copy=True),
            evaluation_user_indices=archive["evaluation_user_indices"].astype(
                np.int32, copy=True
            ),
            fit_matrix=_load_csr(archive, "fit").astype(np.float32),
            seen_matrix=_load_csr(archive, "seen").astype(np.int8),
            targets=archive["targets"].astype(np.int32, copy=True),
            target_timestamps=archive["target_timestamps"].astype(
                np.int64, copy=True
            ),
            cold_users=archive["cold_users"].astype(bool, copy=True),
            archive_sha256=archive_sha,
            protocol_hash=protocol_hash,
            split_artifact_sha256=actual_artifact_hash,
            split_manifest_sha256=canonical_hash(manifest),
            protocol=protocol,
        )

    if data.fit_matrix.shape != data.seen_matrix.shape:
        raise ValueError("fit and seen matrix shapes differ")
    if data.fit_matrix.shape != (data.training_user_ids.size, data.catalog.size):
        raise ValueError("prepared split matrix dimensions are invalid")
    if not (
        data.targets.size
        == data.target_timestamps.size
        == data.evaluation_user_indices.size
        == data.cold_users.size
    ):
        raise ValueError("prepared split evaluation arrays differ in length")
    if manifest.get("dataset") != data.summary():
        raise ValueError("prepared split summary differs from its manifest")
    return data

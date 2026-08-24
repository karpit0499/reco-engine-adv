from __future__ import annotations

import hashlib
import zipfile

import numpy as np
import pytest

from reco_benchmark.dataset import (
    build_global_split,
    build_split_manifest,
    load_prepared_split,
    verify_archive,
    write_split_artifact,
)


def _raw_movie_ids(data, user_id: int, matrix_name: str) -> list[int]:
    user_index = int(np.flatnonzero(data.training_user_ids == user_id)[0])
    matrix = getattr(data, matrix_name)
    return data.catalog.movie_ids[matrix[user_index].indices].tolist()


def test_global_validation_boundary_and_declared_cohorts(miniature_archive) -> None:
    archive, protocol = miniature_archive
    data = build_global_split(archive, protocol, stage="validation")
    assert data.evaluation_user_ids.tolist() == [1, 2]
    assert data.catalog.movie_ids[data.targets].tolist() == [6, 10]
    assert data.target_timestamps.tolist() == [200, 210]
    assert data.cold_users.tolist() == [False, True]
    assert _raw_movie_ids(data, 1, "fit_matrix") == [1, 2, 3, 4, 5]
    assert _raw_movie_ids(data, 1, "seen_matrix") == [1, 2, 3, 4, 5, 12]
    assert 6 not in _raw_movie_ids(data, 1, "fit_matrix")
    assert 12 not in _raw_movie_ids(data, 1, "fit_matrix")
    assert 4 not in data.evaluation_user_ids


def test_global_test_refit_and_equal_timestamp_block(miniature_archive) -> None:
    archive, protocol = miniature_archive
    data = build_global_split(archive, protocol, stage="test")
    assert data.evaluation_user_ids.tolist() == [1, 3]
    assert data.catalog.movie_ids[data.targets].tolist() == [9, 11]
    assert data.target_timestamps.tolist() == [300, 310]
    assert data.cold_users.tolist() == [False, True]
    # Both ratings from timestamp 200 are in the test fitting history; neither was
    # split into validation fitting data.
    assert _raw_movie_ids(data, 1, "fit_matrix") == [1, 2, 3, 4, 5, 6, 7]


def test_prepared_validation_is_physically_separate_from_test(
    miniature_archive, tmp_path
) -> None:
    archive, protocol = miniature_archive
    validation = build_global_split(archive, protocol, stage="validation")
    test = build_global_split(archive, protocol, stage="test")
    validation_path = tmp_path / "validation.npz"
    test_path = tmp_path / "test.npz"
    validation_hash = write_split_artifact(validation_path, validation)
    test_hash = write_split_artifact(test_path, test)
    validation_manifest = build_split_manifest(
        validation, validation_path, validation_hash
    )
    test_manifest = build_split_manifest(test, test_path, test_hash)

    with np.load(validation_path, allow_pickle=False) as payload:
        assert str(payload["stage"][0]) == "validation"
        assert "test_targets" not in payload.files
        assert payload["targets"].tolist() == validation.targets.tolist()
        assert payload["target_timestamps"].max() < 300
    loaded = load_prepared_split(
        validation_path, validation_manifest, protocol, expected_stage="validation"
    )
    assert loaded.summary() == validation.summary()
    assert validation_manifest["artifact_sha256"] != test_manifest["artifact_sha256"]


def test_prepared_artifact_hash_is_mandatory(miniature_archive, tmp_path) -> None:
    archive, protocol = miniature_archive
    data = build_global_split(archive, protocol, stage="validation")
    path = tmp_path / "validation.npz"
    digest = write_split_artifact(path, data)
    manifest = build_split_manifest(data, path, digest)
    manifest["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_prepared_split(path, manifest, protocol, expected_stage="validation")


def test_archive_hash_is_mandatory(miniature_archive) -> None:
    archive, protocol = miniature_archive
    assert verify_archive(archive, protocol) == protocol["dataset"]["archive_sha256"]
    protocol["dataset"]["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_archive(archive, protocol)


def test_duplicate_user_movie_fails(miniature_archive, tmp_path) -> None:
    archive, protocol = miniature_archive
    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(duplicate, "w") as target:
        target.writestr("ml-1m/movies.dat", source.read("ml-1m/movies.dat"))
        ratings = source.read("ml-1m/ratings.dat") + b"1::12::2::180\n"
        target.writestr("ml-1m/ratings.dat", ratings)
    protocol["dataset"]["archive_sha256"] = hashlib.sha256(
        duplicate.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="duplicate user/movie"):
        build_global_split(duplicate, protocol, stage="validation")

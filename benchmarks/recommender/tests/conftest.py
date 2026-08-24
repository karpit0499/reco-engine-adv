from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))


@pytest.fixture()
def miniature_archive(tmp_path: Path) -> tuple[Path, dict]:
    archive = tmp_path / "mini.zip"
    movies = "\n".join(
        f"{movie_id}::Movie {movie_id}::Genre" for movie_id in range(1, 13)
    ) + "\n"
    ratings = "\n".join(
        [
            # Warm user: five positive training events and one negative seen event.
            "1::1::4::100",
            "1::2::5::110",
            "1::3::4::120",
            "1::4::4::130",
            "1::5::5::140",
            "1::12::1::150",
            # Entire timestamp 200 block belongs to validation; movie 6 wins target tie.
            "1::7::4::200",
            "1::6::5::200",
            "1::8::2::220",
            # Entire timestamp 300 block belongs to test.
            "1::9::5::300",
            # Cold validation user.
            "2::10::4::210",
            # Cold test user.
            "3::11::5::310",
            # Sparse-history user is intentionally excluded from validation.
            "4::1::4::100",
            "4::2::4::110",
            "4::3::5::230",
        ]
    ) + "\n"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("ml-1m/movies.dat", movies)
        handle.writestr("ml-1m/ratings.dat", ratings)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    protocol = {
        "schema_version": 2,
        "dataset": {
            "archive_sha256": digest,
            "expected_catalog_items": 12,
            "movies_member": "ml-1m/movies.dat",
            "ratings_member": "ml-1m/ratings.dat",
        },
        "relevance": {"minimum_rating": 4},
        "temporal_split": {
            "train_cutoff_timestamp": 200,
            "test_cutoff_timestamp": 300,
        },
        "cohorts": {
            "warm_minimum_positive_interactions": 5,
            "cold_maximum_observed_interactions": 0,
        },
        "prepared_splits": {
            "validation_artifact": "data/prepared/validation.npz",
            "test_artifact": "data/prepared/test.npz",
        },
    }
    return archive, protocol

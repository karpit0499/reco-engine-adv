"""Read and refresh the sanitized fixture used by the metrics scripts.

The fixture contains only simulator IDs, numeric embeddings, category labels and
aggregate counts. It contains no credentials, raw text, email addresses or other
personal data. Reading it needs only the Python standard library. Refreshing it is
an explicit, read-only GCP operation and needs Application Default Credentials.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT = "reco-engine-adv"
BUCKET = "reco-engine-adv-artifacts"
ITEMS_BLOB = "embeddings/items.json"
CUTOFF = "2026-07-14T09:46:09.789689Z"
CUTOFF_IMPRESSION_ID = "8c4f9fd6-20ac-41a4-ade4-82bc7739c92c"
DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "verified_snapshot.json.gz"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    if fixture.get("schema_version") != 1:
        raise RuntimeError(f"unsupported fixture schema: {fixture.get('schema_version')}")
    return fixture


def fixture_fingerprint(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    try:
        display_path = path.resolve().relative_to(
            Path(__file__).resolve().parents[1]
        ).as_posix()
    except ValueError:
        display_path = path.as_posix()
    return {
        "path": display_path,
        "size_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "json_sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_fixture(fixture: dict[str, Any], path: Path = DEFAULT_FIXTURE) -> None:
    raw = json.dumps(
        fixture, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def refresh_fixture_from_gcp(
    path: Path = DEFAULT_FIXTURE,
    project: str = PROJECT,
    bucket_name: str = BUCKET,
    items_blob_name: str = ITEMS_BLOB,
    cutoff: str = CUTOFF,
    cutoff_impression_id: str = CUTOFF_IMPRESSION_ID,
) -> dict[str, Any]:
    """Read the current synthetic project snapshot and replace the local fixture."""
    from google.cloud import bigquery, storage

    bq = bigquery.Client(project=project)
    gcs = storage.Client(project=project)

    blob = gcs.bucket(bucket_name).blob(items_blob_name)
    blob.reload()
    raw_items = blob.download_as_bytes()
    item_rows = [json.loads(line) for line in raw_items.splitlines() if line.strip()]
    if not item_rows:
        raise RuntimeError(f"gs://{bucket_name}/{items_blob_name} is empty")

    latest = next(iter(bq.query(
        f"SELECT MAX(trained_at) AS trained_at "
        f"FROM `{project}.reco.user_embeddings`"
    ).result()))["trained_at"]
    if latest is None:
        raise RuntimeError("reco.user_embeddings has no training snapshots")

    latest_cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("trained_at", "TIMESTAMP", latest)
    ])
    user_embedding_rows = list(bq.query(
        f"SELECT user_id, embedding FROM `{project}.reco.user_embeddings` "
        "WHERE trained_at = @trained_at ORDER BY user_id",
        job_config=latest_cfg,
    ).result())
    truth = {
        row["user_id"]: row["preferred_category"]
        for row in bq.query(
            f"SELECT user_id, preferred_category FROM `{project}.reco.users_truth`"
        ).result()
    }
    catalog = {
        row["item_id"]: row["category"]
        for row in bq.query(
            f"SELECT item_id, category FROM `{project}.reco.catalog`"
        ).result()
    }

    positive = dict(next(iter(bq.query(f"""
        SELECT
          COUNT(*) AS positive_events,
          COUNT(DISTINCT user_id) AS users,
          COUNT(DISTINCT item_id) AS items,
          COUNT(DISTINCT CONCAT(user_id, CHR(31), item_id)) AS distinct_pairs,
          MIN(event_time) AS first_event_at,
          MAX(event_time) AS last_event_at
        FROM `{project}.reco.events`
        WHERE event_type IN ('click', 'add_to_cart', 'purchase')
    """).result())))

    experiment_cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff),
        bigquery.ScalarQueryParameter(
            "cutoff_impression_id", "STRING", cutoff_impression_id
        ),
    ])
    experiment_rows = list(bq.query(f"""
        WITH eligible AS (
          SELECT * FROM `{project}.reco.impressions`
          WHERE served_at < @cutoff
             OR (served_at = @cutoff AND impression_id <= @cutoff_impression_id)
        ),
        clicks AS (
          SELECT DISTINCT impression_id FROM `{project}.reco.feedback`
          WHERE event = 'click' AND `at` <= @cutoff
        )
        SELECT
          i.user_id,
          ARRAY_AGG(i.variant ORDER BY i.served_at, i.impression_id LIMIT 1)[OFFSET(0)]
            AS variant,
          COUNT(*) AS impressions,
          COUNTIF(c.impression_id IS NOT NULL) AS clicks,
          COUNTIF(i.variant = 'treatment' AND i.served_by = 'popular') AS fallbacks,
          COUNT(DISTINCT i.variant) AS variant_count,
          MAX(i.served_at) AS last_included_at
        FROM eligible i LEFT JOIN clicks c USING (impression_id)
        GROUP BY i.user_id
        ORDER BY i.user_id
    """, job_config=experiment_cfg).result())
    if not experiment_rows or max(row["variant_count"] for row in experiment_rows) != 1:
        raise RuntimeError("experiment assignment is missing or unstable within user")

    all_impressions = next(iter(bq.query(
        f"SELECT COUNT(*) AS n FROM `{project}.reco.impressions`"
    ).result()))["n"]
    per_user = [{
        "user_id": row["user_id"],
        "variant": row["variant"],
        "impressions": int(row["impressions"]),
        "clicks": int(row["clicks"]),
        "fallbacks": int(row["fallbacks"]),
    } for row in experiment_rows]
    eligible_impressions = sum(row["impressions"] for row in per_user)

    item_ids = [str(row["id"]) for row in item_rows]
    user_ids = [row["user_id"] for row in user_embedding_rows if row["user_id"] in truth]
    category_counts = defaultdict(int)
    for category in catalog.values():
        category_counts[category] += 1

    fixture = {
        "schema_version": 1,
        "sanitization": {
            "synthetic_only": True,
            "contains": [
                "simulator user/item IDs",
                "numeric embeddings",
                "synthetic category labels",
                "per-user impression/click/fallback counts",
            ],
            "excludes": [
                "credentials",
                "raw free text",
                "emails",
                "personal data",
                "raw impression IDs",
            ],
        },
        "source": {
            "project": project,
            "item_embeddings": {
                "uri": f"gs://{bucket_name}/{items_blob_name}",
                "generation": str(blob.generation),
                "updated_at": _iso(blob.updated),
                "size_bytes": blob.size,
                "md5_base64": blob.md5_hash,
                "sha256": hashlib.sha256(raw_items).hexdigest(),
            },
            "user_embeddings": {
                "trained_at": _iso(latest),
            },
            "positive_interactions": {
                key: _iso(value) if key.endswith("_at") else int(value)
                for key, value in positive.items()
            },
            "historical_experiment_snapshot": {
                "cutoff": cutoff,
                "cutoff_impression_id_omitted_from_fixture": True,
                "cutoff_provenance": (
                    "final row in the 3,004-impression snapshot reported in repository "
                    "history; not evidence of preregistration"
                ),
                "eligible_impressions": eligible_impressions,
                "all_impressions_at_refresh": int(all_impressions),
                "later_impressions_excluded": int(all_impressions) - eligible_impressions,
                "last_included_at": max(_iso(row["last_included_at"])
                                        for row in experiment_rows),
            },
        },
        "retrieval": {
            "dimension": len(item_rows[0]["embedding"]),
            "item_ids": item_ids,
            "item_categories": [catalog[item_id] for item_id in item_ids],
            "item_embeddings": [row["embedding"] for row in item_rows],
            "user_ids": user_ids,
            "user_preferred_categories": [truth[user_id] for user_id in user_ids],
            "user_embeddings": [
                list(row["embedding"])
                for row in user_embedding_rows
                if row["user_id"] in truth
            ],
            "catalog_category_counts": dict(sorted(category_counts.items())),
        },
        "experiment": {
            "randomization_unit": "user_id",
            "per_user": per_user,
        },
    }
    write_fixture(fixture, path)
    return fixture

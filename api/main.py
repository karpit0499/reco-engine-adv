import os
import json
import uuid
import hashlib
from datetime import datetime, timezone

from flask import Flask, request
from google.cloud import bigquery, aiplatform

app = Flask(__name__)
PROJECT = os.environ["PROJECT"]
REGION = "europe-west3"
INDEX_ENDPOINT = os.environ["INDEX_ENDPOINT"]                 # full resource name
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "items_v1")

aiplatform.init(project=PROJECT, location=REGION)
bq = bigquery.Client(project=PROJECT)

_vs = None


def vs():
    """Lazy so a cold container doesn't pay for the endpoint handshake until
    the first treatment request actually needs it."""
    global _vs
    if _vs is None:
        _vs = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT)
    return _vs


def variant_for(user_id: str) -> str:
    """Deterministic 50/50 split. Hashing the USER (not the request) means a
    user always lands in the same arm — per-user randomization, which is what
    makes the experiment valid. sha256, not hash(): Python's built-in hash is
    salted per process and would reassign users on every container restart."""
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return "treatment" if h % 2 else "control"


def popular_items(k: int = 10):
    q = f"""SELECT item_id FROM `{PROJECT}.reco.item_features_latest`
            ORDER BY popularity_7d DESC LIMIT {k}"""
    return [r["item_id"] for r in bq.query(q).result()]


def user_vector(user_id: str):
    q = f"""SELECT embedding FROM `{PROJECT}.reco.user_embeddings_latest`
            WHERE user_id = @uid"""
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("uid", "STRING", user_id)])
    rows = list(bq.query(q, job_config=cfg).result())
    return list(rows[0]["embedding"]) if rows else None


def retrieve(vec, k: int = 50):
    res = vs().find_neighbors(deployed_index_id=DEPLOYED_INDEX_ID,
                              queries=[vec], num_neighbors=k)
    return [m.id for m in res[0]]


def rank(user_id: str, candidates, k: int = 10):
    q = f"""
    SELECT item_id,
      (SELECT p.prob FROM UNNEST(predicted_label_probs) p WHERE p.label = 1) AS score
    FROM ML.PREDICT(MODEL `{PROJECT}.reco.ranker`, (
      SELECT
        cand AS item_id,
        uf.events_7d, uf.purchases_30d, uf.distinct_items_7d,
        itf.category, itf.price, itf.popularity_7d,
        CAST(uf.top_category = itf.category AS INT64) AS same_category
      FROM UNNEST(@ids) AS cand
      JOIN `{PROJECT}.reco.item_features_latest` itf ON itf.item_id = cand
      CROSS JOIN (SELECT * FROM `{PROJECT}.reco.user_features_latest`
                  WHERE user_id = @uid) uf))
    ORDER BY score DESC
    LIMIT {k}
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("ids", "STRING", candidates),
        bigquery.ScalarQueryParameter("uid", "STRING", user_id)])
    return [r["item_id"] for r in bq.query(q, job_config=cfg).result()]


def log_impression(uid, variant, served_by, items):
    imp_id = str(uuid.uuid4())
    errs = bq.insert_rows_json(f"{PROJECT}.reco.impressions", [{
        "impression_id": imp_id,
        "user_id": uid,
        "variant": variant,
        "served_by": served_by,
        "items": json.dumps(items),
        "served_at": datetime.now(timezone.utc).isoformat(),
    }])
    if errs:
        print("impression insert errors:", errs)
    return imp_id


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/recommend", methods=["POST"])
def recommend():
    body = request.get_json(silent=True) or {}
    uid = body.get("user_id")
    if not uid:
        return {"error": "user_id required"}, 400

    variant = variant_for(uid)
    served_by = "popular"
    items = []

    if variant == "treatment":
        vec = user_vector(uid)
        if vec is not None:
            try:
                items = rank(uid, retrieve(vec, 50), 10)
                served_by = "personalized"
            except Exception as e:
                # Vector Search undeployed, or the ranker model is missing.
                # Fail LOUD: silently serving popular items to the treatment arm
                # would corrupt the experiment with data you can't identify later.
                print("retrieval/ranking failed:", repr(e))
                return {"error": "recommendation backend unavailable"}, 503

    # Control always lands here. So does a treatment user with no embedding yet
    # (a genuine cold start) and a treatment user whose ranking returned nothing.
    # served_by='popular' TAGS those rows so Phase 13 can measure them instead of
    # pretending they didn't happen.
    if not items:
        items = popular_items()
        served_by = "popular"

    imp_id = log_impression(uid, variant, served_by, items)
    return {"impression_id": imp_id, "variant": variant,
            "served_by": served_by, "items": items}, 200


@app.route("/feedback", methods=["POST"])
def feedback():
    """Append-only feedback events. Deliberately NOT an UPDATE on impressions:
    freshly streamed rows sit in BigQuery's streaming buffer for ~90 minutes and
    DML against them fails with `UPDATE or DELETE statement over table would
    affect rows in the streaming buffer, which is not supported`."""
    body = request.get_json(silent=True) or {}
    if body.get("event") not in ("click", "conversion"):
        return {"error": "event must be click|conversion"}, 400
    errs = bq.insert_rows_json(f"{PROJECT}.reco.feedback", [{
        "impression_id": body.get("impression_id"),
        "user_id": body.get("user_id"),
        "event": body["event"],
        "at": datetime.now(timezone.utc).isoformat(),
    }])
    if errs:
        print("feedback insert errors:", errs)
        return {"error": "insert failed"}, 500
    return {"logged": True}, 200
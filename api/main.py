import os
import json
import hashlib
from functools import wraps
from datetime import datetime, timezone

from flask import Flask, request
from google.cloud import bigquery, aiplatform
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

try:
    from .contracts import (
        MAX_JSON_BYTES,
        ContractError,
        ServingVector,
        bearer_auth_status,
        parse_feedback,
        parse_recommend,
        serving_target,
        strip_generation_datapoint,
        validate_active_receipt,
        validate_generation_id,
        verify_vertex_target,
    )
except ImportError:  # Gunicorn imports main.py from /app in the container.
    from contracts import (
        MAX_JSON_BYTES,
        ContractError,
        ServingVector,
        bearer_auth_status,
        parse_feedback,
        parse_recommend,
        serving_target,
        strip_generation_datapoint,
        validate_active_receipt,
        validate_generation_id,
        verify_vertex_target,
    )

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_JSON_BYTES
PROJECT = os.environ.get("PROJECT", "")
REGION = os.environ.get("REGION", "europe-west3")
INDEX_ENDPOINT = os.environ.get("INDEX_ENDPOINT", "")         # full resource name
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "")
INDEX_RESOURCE_NAME = os.environ.get("INDEX_RESOURCE_NAME", "")
API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN")

_bq = None


def bq_client():
    global _bq
    if not PROJECT:
        raise RuntimeError("PROJECT is not configured")
    if _bq is None:
        _bq = bigquery.Client(project=PROJECT)
    return _bq


def vs(snapshot: ServingVector):
    """Return a freshly verified endpoint for one treatment request.

    Endpoint deployments are mutable without changing the configured endpoint
    name or deployed-index id. Re-reading both resources here prevents a silent
    redeploy from routing a captured generation vector to another index.
    """
    if not PROJECT:
        raise RuntimeError("recommendation backend is not configured")
    aiplatform.init(project=PROJECT, location=REGION)
    index = aiplatform.MatchingEngineIndex(
        index_name=snapshot.target.index_name
    )
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=snapshot.target.index_endpoint
    )
    verify_vertex_target(
        index,
        endpoint,
        target=snapshot.target,
        dimensions=len(snapshot.embedding),
    )
    return endpoint


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
    return [r["item_id"] for r in bq_client().query(q).result()]


def user_vector(user_id: str):
    q = f"""SELECT u.embedding, a.generation_id, a.epoch,
                    COUNT(*) OVER () AS matching_rows,
                    (SELECT COUNT(*)
                     FROM `{PROJECT}.reco.active_embedding_generation`
                     WHERE singleton = TRUE) AS active_rows,
                    a.index_name, a.index_endpoint, a.deployed_index_id,
                    a.manifest_uri, a.manifest_object_generation,
                    a.manifest_sha256, a.ops_image_uri, a.source_revision
            FROM `{PROJECT}.reco.active_embedding_generation` a
            JOIN `{PROJECT}.reco.user_embedding_generations` u
              ON u.generation_id = a.generation_id
            WHERE a.singleton = TRUE AND u.user_id = @uid
            QUALIFY ROW_NUMBER() OVER (ORDER BY u.trained_at DESC) = 1"""
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("uid", "STRING", user_id)])
    rows = list(bq_client().query(q, job_config=cfg).result())
    if not rows:
        return None
    row = rows[0]
    if int(row["active_rows"]) != 1 or int(row["matching_rows"]) != 1:
        raise RuntimeError("active generation/user vector is not a singleton")
    configured = serving_target(
        INDEX_RESOURCE_NAME, INDEX_ENDPOINT, DEPLOYED_INDEX_ID
    )
    active_target = serving_target(
        row["index_name"], row["index_endpoint"], row["deployed_index_id"]
    )
    if active_target != configured:
        raise RuntimeError("active generation is bound to another Vertex target")
    generation = validate_generation_id(str(row["generation_id"]))
    validate_active_receipt(
        generation,
        manifest_uri=row["manifest_uri"],
        manifest_object_generation=row["manifest_object_generation"],
        manifest_sha256=row["manifest_sha256"],
        ops_image_uri=row["ops_image_uri"],
        source_revision=row["source_revision"],
    )
    epoch = int(row["epoch"])
    if epoch <= 0:
        raise RuntimeError("active generation epoch is invalid")
    return ServingVector(
        embedding=tuple(float(value) for value in row["embedding"]),
        generation_id=generation,
        epoch=epoch,
        target=active_target,
    )


def retrieve(snapshot: ServingVector, k: int = 50):
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
        Namespace,
    )

    res = vs(snapshot).find_neighbors(
        deployed_index_id=snapshot.target.deployed_index_id,
        queries=[list(snapshot.embedding)],
        num_neighbors=k,
        filter=[Namespace("generation", [snapshot.generation_id], [])],
    )
    neighbors = res[0] if res else []
    return [
        strip_generation_datapoint(snapshot.generation_id, match.id)
        for match in neighbors
    ]


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
    return [r["item_id"] for r in bq_client().query(q, job_config=cfg).result()]


def log_impression(uid, variant, served_by, items, impression_id):
    errs = bq_client().insert_rows_json(f"{PROJECT}.reco.impressions", [{
        "impression_id": impression_id,
        "user_id": uid,
        "variant": variant,
        "served_by": served_by,
        "items": json.dumps(items),
        "served_at": datetime.now(timezone.utc).isoformat(),
    }], row_ids=[impression_id])
    if errs:
        raise RuntimeError("impression insert failed")
    return impression_id


def impression_belongs_to_user(impression_id, user_id):
    q = f"""SELECT 1 FROM `{PROJECT}.reco.impressions`
            WHERE impression_id = @impression_id AND user_id = @user_id
            LIMIT 1"""
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("impression_id", "STRING", impression_id),
        bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
    ])
    return next(iter(bq_client().query(q, job_config=cfg).result()), None) is not None


def log_feedback(event):
    errs = bq_client().insert_rows_json(f"{PROJECT}.reco.feedback", [{
        "impression_id": event.impression_id,
        "user_id": event.user_id,
        "event": event.event,
        "at": datetime.now(timezone.utc).isoformat(),
    }], row_ids=[event.event_id])
    if errs:
        raise RuntimeError("feedback insert failed")


def require_api_bearer(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        status = bearer_auth_status(API_BEARER_TOKEN, request.headers.get("Authorization"))
        if status == "misconfigured":
            app.logger.error("API bearer authentication is not safely configured")
            return {"error": "service unavailable"}, 503
        if status != "ok":
            return {"error": "unauthorized"}, 401, {"WWW-Authenticate": "Bearer"}
        return view(*args, **kwargs)
    return wrapped


def request_json():
    if not request.is_json:
        raise UnsupportedMediaType()
    body = request.get_json(silent=False)
    if not isinstance(body, dict):
        raise ContractError("request body must be a JSON object")
    return body


@app.errorhandler(RequestEntityTooLarge)
def body_too_large(_error):
    return {"error": "request body too large"}, 413


@app.errorhandler(UnsupportedMediaType)
def wrong_media_type(_error):
    return {"error": "Content-Type must be application/json"}, 415


@app.errorhandler(BadRequest)
def malformed_json(_error):
    return {"error": "malformed JSON"}, 400


@app.after_request
def response_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/recommend", methods=["POST"])
@require_api_bearer
def recommend():
    try:
        recommendation_request = parse_recommend(request_json())
    except ContractError as error:
        return {"error": str(error)}, 400
    uid = recommendation_request.user_id

    try:
        variant = variant_for(uid)
        served_by = "popular"
        items = []

        if variant == "treatment":
            snapshot = user_vector(uid)
            if snapshot is not None:
                items = rank(uid, retrieve(snapshot, 50), 10)
                served_by = "personalized"

        # Control always lands here. So does a treatment user with no embedding
        # or no ranked results. served_by preserves that distinction downstream.
        if not items:
            items = popular_items()
            served_by = "popular"
    except Exception:
        app.logger.error("recommendation generation failed")
        return {"error": "recommendation backend unavailable"}, 503

    try:
        impression_id = log_impression(
            uid,
            variant,
            served_by,
            items,
            recommendation_request.request_id,
        )
    except Exception:
        # Never return recommendations that are absent from experiment logging.
        app.logger.error("impression logging failed")
        return {"error": "recommendation logging unavailable"}, 503

    return {"impression_id": impression_id, "variant": variant,
            "served_by": served_by, "items": items}, 200


@app.route("/feedback", methods=["POST"])
@require_api_bearer
def feedback():
    """Append-only feedback events. Deliberately NOT an UPDATE on impressions:
    freshly streamed rows sit in BigQuery's streaming buffer for ~90 minutes and
    DML against them fails with `UPDATE or DELETE statement over table would
    affect rows in the streaming buffer, which is not supported`."""
    try:
        event = parse_feedback(request_json())
    except ContractError as error:
        return {"error": str(error)}, 400

    try:
        linked = impression_belongs_to_user(event.impression_id, event.user_id)
    except Exception:
        app.logger.error("feedback linkage lookup failed")
        return {"error": "feedback backend unavailable"}, 503
    if not linked:
        return {"error": "impression and user are not linked"}, 409

    try:
        log_feedback(event)
    except Exception:
        app.logger.error("feedback logging failed")
        return {"error": "feedback logging unavailable"}, 503
    return {"event_id": event.event_id, "logged": True}, 200

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # MUST precede the tensorflow import

import argparse
import json
import sys
from datetime import datetime, timezone

import tensorflow as tf
import tensorflow_recommenders as tfrs
from google.cloud import bigquery, storage

from generation_contract import (
    USER_GENERATIONS_TABLE,
    build_manifest,
    canonical_item_payload,
    canonical_user_payload,
    item_blob_name,
    manifest_blob_name,
    user_blob_name,
    validate_binding,
    validate_generation_id,
    validate_ops_image_uri,
    validate_source_revision,
)

PROJECT = os.environ["PROJECT"]
BUCKET = os.environ.get("ARTIFACT_BUCKET", f"{PROJECT}-artifacts")
VECTOR_BUCKET = os.environ.get("VECTOR_BUCKET", f"{PROJECT}-vectors")
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--generation-id", default=os.environ.get("GENERATION_ID"))
argument_parser.add_argument("--index-name", default=os.environ.get("INDEX_RESOURCE_NAME"))
argument_parser.add_argument("--index-endpoint", default=os.environ.get("INDEX_ENDPOINT"))
argument_parser.add_argument(
    "--deployed-index-id", default=os.environ.get("DEPLOYED_INDEX_ID")
)
argument_parser.add_argument("--ops-image-uri", default=os.environ.get("RECO_OPS_IMAGE"))
argument_parser.add_argument("--source-revision", default=os.environ.get("SOURCE_REVISION"))
arguments = argument_parser.parse_args()
if not arguments.generation_id:
    argument_parser.error("--generation-id or GENERATION_ID is required")
GENERATION_ID = validate_generation_id(arguments.generation_id)
BINDING = validate_binding(
    arguments.index_name, arguments.index_endpoint, arguments.deployed_index_id
)
OPS_IMAGE_URI = validate_ops_image_uri(arguments.ops_image_uri)
SOURCE_REVISION = validate_source_revision(arguments.source_revision)
MODEL_ROOT = os.environ.get("AIP_MODEL_DIR", f"gs://{BUCKET}/models/twotower")
MODEL_DIR = f"{MODEL_ROOT.rstrip('/')}/generations/{GENERATION_ID}"
ITEMS_BLOB = item_blob_name(GENERATION_ID)
USERS_BLOB = user_blob_name(GENERATION_ID)
MANIFEST_BLOB = manifest_blob_name(GENERATION_ID)
EMB = 32                                 # MUST equal `dimensions` in Phase 8's index

bq = bigquery.Client(project=PROJECT)
gcs = storage.Client(project=PROJECT)

# A completed manifest is the write-once commit marker for this id. Reusing it
# would make an evaluation receipt ambiguous, so fail before training or writes.
if gcs.bucket(BUCKET).blob(MANIFEST_BLOB).exists():
    sys.exit(f"generation {GENERATION_ID!r} already has a completed manifest")


def upload_immutable(bucket_name, blob_name, payload, content_type):
    blob = gcs.bucket(bucket_name).blob(blob_name)
    try:
        blob.upload_from_string(
            payload,
            content_type=content_type,
            if_generation_match=0,
        )
    except Exception:
        blob.reload()
        generation = int(blob.generation)
        existing = blob.download_as_bytes(if_generation_match=generation)
        if existing != payload:
            raise
    blob.reload()
    return int(blob.generation)

# ── Positives: any click / add_to_cart / purchase is an interaction ──────
pairs = bq.query(f"""
    SELECT user_id, item_id
    FROM `{PROJECT}.reco.events`
    WHERE event_type IN ('click', 'add_to_cart', 'purchase')
""").to_dataframe()

if len(pairs) < 5000:
    sys.exit(f"only {len(pairs)} interaction pairs — let the generator and the "
             f"Dataflow job run longer. 5,000 is the hard floor; 20,000+ is what you want, "
             f"which is roughly 90 minutes of the Phase 2 generator at SLEEP=0.1.")
print(f"{len(pairs)} positive interaction pairs")

# .astype(str) is defensive: BigQuery's Arrow path can hand back nullable/extension
# string dtypes that tf.data refuses to convert into a tf.string tensor.
users = pairs["user_id"].astype(str)
items = pairs["item_id"].astype(str)
user_ids = sorted(users.unique().tolist())
item_ids = sorted(items.unique().tolist())
# NOTE: item_ids comes from INTERACTIONS, not from the catalog. An item nobody has ever
# clicked gets no embedding, so it never enters the Vector Search index and can never be
# retrieved. That is a real cold-start property of two-tower retrieval, not a bug — but it
# means len(item_ids) can be < 500, and Phase 19's export has to allow for that.
print(f"{len(user_ids)} users, {len(item_ids)} items in the training set "
      f"(catalog is 500 — the difference is items with zero positive interactions)")

ds = (tf.data.Dataset.from_tensor_slices({
        "user_id": users.to_numpy(),
        "item_id": items.to_numpy()})
      .shuffle(200_000, seed=42)
      .batch(512)
      .prefetch(tf.data.AUTOTUNE))
items_ds = tf.data.Dataset.from_tensor_slices(item_ids)


def tower(vocab):
    # StringLookup reserves index 0 for out-of-vocabulary, so the embedding
    # table needs len(vocab) + 1 rows.
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None),
        tf.keras.layers.Embedding(len(vocab) + 1, EMB),
    ])


user_model, item_model = tower(user_ids), tower(item_ids)


class TwoTower(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items_ds.batch(256).map(item_model)))

    def compute_loss(self, feats, training=False):
        return self.task(self.user_model(feats["user_id"]),
                         self.item_model(feats["item_id"]))


model = TwoTower()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(ds, epochs=5)     # watch factorized_top_k/top_100_categorical_accuracy climb

# ── Export the towers with an explicit string-input signature ────────────
# Nothing in this build SERVES these SavedModels (the API uses precomputed
# vectors). They're exported so that "deploy a real-time user tower" is a
# one-liner later — and because an untested export is a lie in your README.
class Exporter(tf.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def __call__(self, ids):
        return self.m(ids)


tf.saved_model.save(Exporter(user_model), f"{MODEL_DIR}/user_tower")
tf.saved_model.save(Exporter(item_model), f"{MODEL_DIR}/item_tower")
print("saved towers →", MODEL_DIR)

# Immutable candidate items → canonical and regional GCS.
vecs = item_model(tf.constant(item_ids)).numpy()
raw_item_rows = [
    {"id": iid, "embedding": [float(x) for x in vector]}
    for iid, vector in zip(item_ids, vecs)
]
items_payload, item_rows = canonical_item_payload(raw_item_rows, dimensions=EMB)
item_object_generations = {}
for bucket_name in (BUCKET, VECTOR_BUCKET):
    item_object_generations[bucket_name] = upload_immutable(
        bucket_name,
        ITEMS_BLOB,
        items_payload,
        "application/x-ndjson",
    )
    print(f"staged {len(item_ids)} item embeddings → gs://{bucket_name}/{ITEMS_BLOB}")

# Candidate users → generation-keyed staging, never the live view.
uvecs = user_model(tf.constant(user_ids)).numpy()
now = datetime.now(timezone.utc).isoformat()
raw_user_rows = [
    {
        "user_id": user_id,
        "embedding": [float(x) for x in vector],
    }
    for user_id, vector in zip(user_ids, uvecs)
]
users_payload, canonical_users = canonical_user_payload(raw_user_rows, dimensions=EMB)
user_object_generation = upload_immutable(
    BUCKET,
    USERS_BLOB,
    users_payload,
    "application/x-ndjson",
)
user_rows = [
    dict(row, generation_id=GENERATION_ID, trained_at=now)
    for row in canonical_users
]
user_table_id = f"{PROJECT}.reco.{USER_GENERATIONS_TABLE}"
user_table = bigquery.Table(
    user_table_id,
    schema=[
        bigquery.SchemaField("generation_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
        bigquery.SchemaField("trained_at", "TIMESTAMP", mode="REQUIRED"),
    ],
)
bq.create_table(user_table, exists_ok=True)
load_config = bigquery.LoadJobConfig(
    schema=user_table.schema,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
)
job_id = f"stage_user_embeddings_{GENERATION_ID.replace('.', '_').replace('-', '_')}"
bq.load_table_from_json(
    user_rows,
    user_table_id,
    job_config=load_config,
    job_id=job_id,
).result()
print(f"staged {len(user_rows)} user embeddings → {user_table_id}")

# The manifest is the candidate's commit marker and is uploaded last.
manifest = build_manifest(
    generation_id=GENERATION_ID,
    created_at=now,
    trained_at=now,
    dimensions=EMB,
    items_payload=items_payload,
    item_count=len(item_rows),
    canonical_bucket=BUCKET,
    canonical_item_generation=item_object_generations[BUCKET],
    regional_bucket=VECTOR_BUCKET,
    regional_item_generation=item_object_generations[VECTOR_BUCKET],
    users_payload=users_payload,
    user_count=len(user_rows),
    user_object_generation=user_object_generation,
    model_uri=MODEL_DIR,
    project=PROJECT,
    binding=BINDING,
    ops_image_uri=OPS_IMAGE_URI,
    source_revision=SOURCE_REVISION,
    canary_item_id=item_rows[0]["id"],
    canary_user_id=canonical_users[0]["user_id"],
)
gcs.bucket(BUCKET).blob(MANIFEST_BLOB).upload_from_string(
    json.dumps(manifest, indent=2, sort_keys=True),
    content_type="application/json",
    if_generation_match=0,
)
print(f"completed generation manifest → gs://{BUCKET}/{MANIFEST_BLOB}")

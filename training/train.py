import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # MUST precede the tensorflow import

import json
import sys
from datetime import datetime, timezone

import tensorflow as tf
import tensorflow_recommenders as tfrs
from google.cloud import bigquery, storage

PROJECT = os.environ["PROJECT"]
BUCKET = "reco-engine-adv-artifacts"
MODEL_DIR = os.environ.get("AIP_MODEL_DIR",
                           f"gs://{BUCKET}/models/twotower/dev")
ITEMS_BLOB = "embeddings/items.json"     # Vector Search reads gs://BUCKET/embeddings/
EMB = 32                                 # MUST equal `dimensions` in Phase 8's index

bq = bigquery.Client(project=PROJECT)

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

# ── Item embeddings → GCS JSONL (the Vector Search ingestion format) ─────
vecs = item_model(tf.constant(item_ids)).numpy()
payload = "\n".join(
    json.dumps({"id": iid, "embedding": [float(x) for x in v]})
    for iid, v in zip(item_ids, vecs)
) + "\n"
storage.Client(project=PROJECT).bucket(BUCKET).blob(ITEMS_BLOB).upload_from_string(payload)
print(f"wrote {len(item_ids)} item embeddings → gs://{BUCKET}/{ITEMS_BLOB}")

# ── User embeddings → BigQuery (APPEND with trained_at; the API reads the
#    _latest view. Never TRUNCATE+insert: the streaming buffer blocks DML
#    on freshly-streamed rows for ~90 minutes.) ────────────────────────────
uvecs = user_model(tf.constant(user_ids)).numpy()
now = datetime.now(timezone.utc).isoformat()
rows = [{"user_id": u, "embedding": [float(x) for x in v], "trained_at": now}
        for u, v in zip(user_ids, uvecs)]
for i in range(0, len(rows), 500):
    errs = bq.insert_rows_json(f"{PROJECT}.reco.user_embeddings", rows[i:i + 500])
    if errs:
        raise RuntimeError(errs)
print(f"wrote {len(rows)} user embeddings → reco.user_embeddings")
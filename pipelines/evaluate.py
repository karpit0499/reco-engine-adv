"""Offline recall gate. Scores the FRESH embedding space (new item vectors from GCS
x new user vectors from BigQuery) by brute force — deliberately never touching the
live index, whose item vectors are still from the previous run."""
import argparse
import json
import os
import sys

import numpy as np
from google.cloud import bigquery, storage

PROJECT = os.environ["PROJECT"]
BUCKET = "reco-engine-adv-artifacts"          # EU multi-region — canonical
ITEMS_BLOB = "embeddings/items.json"
CHAMPION_BLOB = "eval/champion.json"          # the reigning score, written on promote
TOP_K = 10
N_USERS = 50

FLOOR = float(os.environ.get("RECALL_FLOOR", "0.35"))    # absolute minimum; 0.20 = chance
MARGIN = float(os.environ.get("RECALL_MARGIN", "0.03"))  # allowed regression vs champion

ap = argparse.ArgumentParser()
ap.add_argument("--decision-out", required=True)
ap.add_argument("--score-out", required=True)
args = ap.parse_args()

bq = bigquery.Client(project=PROJECT)
gcs = storage.Client(project=PROJECT)
bucket = gcs.bucket(BUCKET)

# ── 1. New item vectors, straight from the file train.py just wrote ──────────
rows = [json.loads(l) for l in
        bucket.blob(ITEMS_BLOB).download_as_text().splitlines() if l.strip()]
if not rows:
    sys.exit(f"gs://{BUCKET}/{ITEMS_BLOB} is empty — training produced nothing")

item_ids = [r["id"] for r in rows]
item_mat = np.array([r["embedding"] for r in rows], dtype=np.float32)  # (N, 32)
print(f"{len(item_ids)} item vectors, dim {item_mat.shape[1]}")

# ── 2. New user vectors + ground truth ──────────────────────────────────────
uvecs = {r["user_id"]: np.array(r["embedding"], dtype=np.float32) for r in bq.query(
    f"SELECT user_id, embedding FROM `{PROJECT}.reco.user_embeddings_latest`").result()}
truth = {r["user_id"]: r["preferred_category"] for r in bq.query(
    f"SELECT user_id, preferred_category FROM `{PROJECT}.reco.users_truth`").result()}
catalog = {r["item_id"]: r["category"] for r in bq.query(
    f"SELECT item_id, category FROM `{PROJECT}.reco.catalog`").result()}

if not uvecs:
    sys.exit("no user embeddings — train.py did not write to reco.user_embeddings")

# ── 3. Brute-force top-K. DOT_PRODUCT, to match the index's distance measure ─
sample = sorted(u for u in uvecs if u in truth)[:N_USERS]
hits = total = 0
for uid in sample:
    scores = item_mat @ uvecs[uid]                       # (N,)
    top = np.argsort(-scores)[:TOP_K]                    # highest dot product first
    cats = [catalog.get(item_ids[i], "?") for i in top]
    hits += sum(c == truth[uid] for c in cats)
    total += TOP_K

score = hits / total
print(f"\npreferred-category share in top-{TOP_K}: {score:.1%}  "
      f"(chance = 20%, floor = {FLOOR:.0%})")

# ── 4. Compare against the champion ─────────────────────────────────────────
champ_blob = bucket.blob(CHAMPION_BLOB)
champion = None
if champ_blob.exists():
    champion = float(json.loads(champ_blob.download_as_text())["score"])
    print(f"champion score: {champion:.1%}")
else:
    print("no champion on record — this run will set the baseline")

if score < FLOOR:
    decision, why = "hold", f"{score:.1%} is below the {FLOOR:.0%} floor"
elif champion is not None and score < champion - MARGIN:
    decision, why = "hold", (f"{score:.1%} regressed more than {MARGIN:.0%} "
                             f"below champion {champion:.1%}")
else:
    decision, why = "promote", "no regression, above floor"

print(f"\nDECISION: {decision.upper()} — {why}")

if decision == "promote":
    champ_blob.upload_from_string(json.dumps({"score": score}))
    print(f"champion updated → {score:.1%}")

# ── 5. Hand the results back to KFP ─────────────────────────────────────────
for path, value in ((args.decision_out, decision), (args.score_out, str(score))):
    os.makedirs(os.path.dirname(path), exist_ok=True)   # KFP does not always pre-create it
    with open(path, "w") as f:
        f.write(value)
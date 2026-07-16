import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import requests
import xgboost as xgb
from google.cloud import bigquery, storage

PROJECT = os.environ.get("PROJECT", "reco-engine-adv")
REGION = "europe-west3"
BUCKET = f"{PROJECT}-artifacts"
OUT = pathlib.Path(__file__).resolve().parent.parent / "demo" / "data"
OUT.mkdir(parents=True, exist_ok=True)

bq = bigquery.Client(project=PROJECT)
gcs = storage.Client(project=PROJECT)
q = lambda sql: list(bq.query(sql).result())
flat = lambda A: [round(float(v), 4) for v in np.asarray(A).ravel()]


def project2(X):
    """32-d -> 2-d, for looking at. t-SNE, not PCA: five near-orthogonal category centroids
    in 32-d cannot all be separated by two principal components, and two of your categories
    end up drawn on top of each other. Say 't-SNE' on the panel — screen distance is not
    dot-product distance, and the retrieval maths never touches these coordinates."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=1,
                max_iter=800).fit_transform(np.asarray(X)).astype(np.float32)


# ── 1. Catalogue + item features ─────────────────────────────────────────────
print("· catalogue + item features")
items = q(f"""
  SELECT item_id, category, price, popularity_7d
  FROM `{PROJECT}.reco.item_features_latest`      -- _latest, never the raw table (fan-out trap)
  ORDER BY item_id
""")
item_ids = [r["item_id"] for r in items]
cats = sorted({r["category"] for r in items})
print(f"  {len(item_ids)} items, {len(cats)} categories")

# ── 2. Item vectors: the current run from GCS, older runs from the archive ───
print("· item vectors")


def read_item_vectors(blob_name):
    blob = gcs.bucket(BUCKET).blob(blob_name)
    if not blob.exists():
        return None
    by_id = {}
    for line in blob.download_as_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_id[r["id"]] = r["embedding"]
    return by_id


cur = read_item_vectors("embeddings/items.json")
if not cur:
    sys.exit(f"gs://{BUCKET}/embeddings/items.json is empty — run training/train.py (Phase 7)")
DIM = len(next(iter(cur.values())))
print(f"  dim = {DIM}")

# Older runs only exist if you patched train.py to archive them (see Phase 19, Step 19.1).
archive = sorted(b.name for b in gcs.list_blobs(BUCKET, prefix="archive/items_"))
prev = read_item_vectors(archive[-1]) if archive else None
print(f"  archived runs: {len(archive)}" + (f" (using {archive[-1]})" if archive else " — retrain demo will be disabled"))

# ⚠️ item_ids came from item_features_latest = ALL 500 catalog items. `cur` came from
# train.py, whose item vocabulary is built from items that were actually INTERACTED with.
# An item nobody ever clicked has no embedding — so indexing cur[i] blindly is a KeyError
# waiting for the day you train on a short event log. Intersect, and say so out loud.
missing = [i for i in item_ids if i not in cur]
if missing:
    print(f"  ! {len(missing)} catalogue items have no embedding (zero positive "
          f"interactions) — they are not in the index and cannot be retrieved. Dropping "
          f"them from the demo bundle. e.g. {missing[:5]}")
    keep = set(cur)
    items = [r for r in items if r["item_id"] in keep]
    item_ids = [i for i in item_ids if i in keep]
if not item_ids:
    sys.exit("no catalogue item has an embedding — did training/train.py actually run?")

I1 = np.array([cur[i] for i in item_ids], dtype=np.float32)
I2 = (np.array([prev[i] for i in item_ids], dtype=np.float32)
      if prev and all(i in prev for i in item_ids) else None)

# ── 3. User vectors: two most recent trained_at snapshots (append-only pays off) ──
print("· user vectors + features + ground truth")
runs = [r["trained_at"] for r in q(f"""
  SELECT DISTINCT trained_at FROM `{PROJECT}.reco.user_embeddings`
  ORDER BY trained_at DESC LIMIT 2
""")]
if not runs:
    sys.exit("reco.user_embeddings is empty — run training/train.py (Phase 7)")


def user_run(ts):
    return {r["user_id"]: list(r["embedding"]) for r in q(f"""
      SELECT user_id, embedding FROM `{PROJECT}.reco.user_embeddings`
      WHERE trained_at = TIMESTAMP('{ts.isoformat()}')
    """)}


U_cur = user_run(runs[0])
U_prev = user_run(runs[1]) if len(runs) > 1 else None

feats = q(f"""
  SELECT uf.user_id, uf.events_7d, uf.purchases_30d, uf.distinct_items_7d, uf.top_category,
         ut.preferred_category
  FROM `{PROJECT}.reco.user_features_latest` uf
  JOIN `{PROJECT}.reco.users_truth` ut USING (user_id)
  ORDER BY uf.user_id
""")
user_ids = [r["user_id"] for r in feats if r["user_id"] in U_cur]
F = {r["user_id"]: r for r in feats}
print(f"  {len(user_ids)} users with an embedding and a feature row")

U1 = np.array([U_cur[u] for u in user_ids], dtype=np.float32)
U2 = np.array([U_prev[u] for u in user_ids], dtype=np.float32) if (
    U_prev and all(u in U_prev for u in user_ids)) else None

if (I2 is None) != (U2 is None):
    # The asymmetry is worth a beat: user vectors are versioned for free because
    # reco.user_embeddings is APPEND-ONLY, while items.json in GCS is overwritten on every
    # run. Same lesson, two storage layers. Archive the blob (Phase 19) and it goes away.
    print("  ! only one half of a second run is available — Lab retrain stays disabled")
    I2 = U2 = None

item_payload = {
    "dim": DIM, "categories": cats, "ids": item_ids,
    "category": [r["category"] for r in items],
    "price": [float(r["price"]) for r in items],
    "popularity_7d": [int(r["popularity_7d"]) for r in items],
    "emb": flat(I1), "xy": flat(project2(I1)),
}
if I2 is not None:
    item_payload |= {"emb_v2": flat(I2), "xy_v2": flat(project2(I2))}
json.dump(item_payload, open(OUT / "items.json", "w"))

user_payload = {
    "ids": user_ids,
    "truth": [F[u]["preferred_category"] for u in user_ids],
    "emb": flat(U1),
    "feat": {
        "events_7d": [int(F[u]["events_7d"]) for u in user_ids],
        "purchases_30d": [int(F[u]["purchases_30d"]) for u in user_ids],
        "distinct_items_7d": [int(F[u]["distinct_items_7d"]) for u in user_ids],
        "top_category": [F[u]["top_category"] for u in user_ids],
    },
}
if U2 is not None:
    user_payload["emb_v2"] = flat(U2)
json.dump(user_payload, open(OUT / "users.json", "w"))

# ── 4. The ranker ────────────────────────────────────────────────────────────
# BQML's BOOSTED_TREE_CLASSIFIER *is* XGBoost, so it exports as an XGBoost booster.
# Two things this code refuses to do:
#   1) trust get_dump(dump_format="json") — it ROUNDS split_condition, and rows that sit on a
#      split boundary then take the wrong branch. Silent. Use save_model()'s flat arrays.
#   2) ship without checking. Every exported ranker is scored against BQML's own ML.PREDICT
#      before it is written to disk. An untested export is a lie in your README.
print("· ranker")
# BQML BOOSTED_TREE_CLASSIFIER = XGBoost multi:softprob, 2 classes (verified 7e-08 vs ML.PREDICT):
#   • leaf value lives in split_conditions ("c"); base_weights is the un-shrunk weight — wrong array.
#   • category is LABEL-encoded to float(vocab.index(value)) — NOT one-hot. (See predictor.py in the tarball.)
#   • score = sum trees BY CLASS GROUP (tree_info), softmax, take P(label=1). Summing all trees
#     + one sigmoid makes the two class groups cancel to 0 → 0.5 on every row (that was the 0.314).
import json, math, pathlib, tempfile, subprocess, sys
import numpy as np, xgboost as xgb
from google.cloud import bigquery, storage
f32 = np.float32

tmp = pathlib.Path(tempfile.mkdtemp())
subprocess.run(["bq", "extract", "--destination_format=ML_XGBOOST_BOOSTER",
                "-m", f"{PROJECT}:reco.ranker", f"gs://{BUCKET}/export/ranker"], check=True)
bkt = storage.Client(project=PROJECT).bucket(BUCKET)
bkt.blob("export/ranker/model.bst").download_to_filename(str(tmp / "model.bst"))
meta = json.loads(bkt.blob("export/ranker/assets/model_metadata.json").download_as_text())

booster = xgb.Booster(); booster.load_model(str(tmp / "model.bst"))
booster.save_model(str(tmp / "ranker.json"))
raw = json.load(open(tmp / "ranker.json"))
learner = raw["learner"]; gbm = learner["gradient_booster"]["model"]

num_class = int(learner["learner_model_param"].get("num_class", "0") or "0")
tree_info = list(gbm.get("tree_info", [0] * len(gbm["trees"])))
n_groups  = max(num_class, max(tree_info) + 1, 1)
class_names = [str(c) for c in meta.get("class_names", ["1", "0"])]
pos_group = class_names.index("1")   # which softmax group is P(label=1)

# leaf value in split_conditions ("c"); "d" = default_left, for routing missing values.
trees = [{"l": t["left_children"], "r": t["right_children"], "i": t["split_indices"],
          "c": [float(v) for v in t["split_conditions"]], "d": t["default_left"]}
         for t in gbm["trees"]]

FEATS = meta["feature_names"]
CAT = [f for f in FEATS if meta["features"].get(f, {}).get("encode_type") == "ohe"][0]
cat_idx = FEATS.index(CAT)
vocab = bkt.blob(f"export/ranker/assets/{cat_idx}_categorical_label.txt").download_as_text().splitlines()

def encode(r):
    out = []
    for f in FEATS:
        if f == CAT:
            v = str(r[f]); out.append(float(vocab.index(v)) if v in vocab else None)  # unseen -> missing
        else:
            out.append(float(r[f]))
    return out

def walk(t, row):
    i = 0
    while t["l"][i] != -1:
        x = row[t["i"][i]]
        go = (t["d"][i] == 1) if x is None else (f32(x) < f32(t["c"][i]))
        i = t["l"][i] if go else t["r"][i]
    return t["c"][i]

def score(row):
    m = [0.0] * n_groups
    for k, t in enumerate(trees):
        m[tree_info[k]] += walk(t, row)
    mx = max(m); e = [math.exp(v - mx) for v in m]; s = sum(e)
    return e[pos_group] / s

# THE GATE — 200 rows BigQuery already scored with ML.PREDICT are ground truth.
print("  verifying against ML.PREDICT ...")
gold = list(bigquery.Client(project=PROJECT).query(f"""
  SELECT {', '.join(FEATS)},
         (SELECT p.prob FROM UNNEST(predicted_label_probs) p WHERE p.label = 1) AS prob
  FROM ML.PREDICT(MODEL `{PROJECT}.reco.ranker`,
                  (SELECT * EXCEPT(label) FROM `{PROJECT}.reco.rank_training` LIMIT 200))
""").result())
worst = max(abs(score(encode(dict(r))) - float(r["prob"])) for r in gold)
print(f"  max |browser − ML.PREDICT| = {worst:.2e}")
if worst > 1e-4:
    sys.exit("ranker export does NOT reproduce BQML. Do not ship it.")

# Ship exactly what the browser needs to reproduce this (label-encode + per-group softmax).
OUT = pathlib.Path("demo/data"); OUT.mkdir(parents=True, exist_ok=True)
json.dump({"features": FEATS, "categorical": CAT, "cat_index": cat_idx, "vocab": vocab,
           "trees": trees, "tree_info": tree_info, "n_groups": n_groups, "pos_group": pos_group},
          open(OUT / "ranker.json", "w"))
print(f"  ✓ wrote demo/data/ranker.json — {len(trees)} trees, {n_groups} class groups, residual {worst:.1e}")
print("· golden set from the live API")
api = subprocess.run(["gcloud", "run", "services", "describe", "reco-api",
                      "--region", REGION, "--format=value(status.url)"],
                     capture_output=True, text=True).stdout.strip()
golden_api = []
if api:
    for u in user_ids[:25]:
        try:
            r = requests.post(f"{api}/recommend", json={"user_id": u}, timeout=30)
            body = r.json()
            if body.get("served_by") == "personalized":
                golden_api.append({"user_id": u, "items": body["items"]})
        except Exception as e:
            print("   !", u, repr(e))
    print(f"  {len(golden_api)} personalised responses recorded")
else:
    print("  ! reco-api not found — the Fidelity panel will say so, which is the honest outcome")
# ranker golden — 20 rows the browser must reproduce, with BQML's own prob alongside for the fidelity panel
golden_ranker = [{"row": {f: (str(dict(r)[f]) if f == CAT else float(dict(r)[f])) for f in FEATS},
                  "browser": score(encode(dict(r))), "ml_predict": float(r["prob"])}
                 for r in gold[:20]]
json.dump({"api": golden_api, "ranker": golden_ranker}, open(OUT / "golden.json", "w"))

# ── 6. The experiment ────────────────────────────────────────────────────────
print("· experiment")
topline = q(f"""
  WITH clicks AS (SELECT DISTINCT impression_id FROM `{PROJECT}.reco.feedback` WHERE event='click')
  SELECT i.variant, COUNT(*) n, COUNTIF(c.impression_id IS NOT NULL) clicks,
         COUNTIF(i.served_by='popular') fallbacks
  FROM `{PROJECT}.reco.impressions` i LEFT JOIN clicks c USING (impression_id)
  GROUP BY variant
""")
d = {r["variant"]: dict(r) for r in topline}
exp = {"shipped": {k: {kk: int(vv) for kk, vv in v.items() if kk != "variant"} for k, v in d.items()}}
if "treatment" in d and "control" in d and d["control"]["n"]:
    t, c = d["treatment"], d["control"]
    ctr_t, ctr_c = t["clicks"] / t["n"], c["clicks"] / c["n"]
    exp["headline_lift"] = f"{ctr_t / ctr_c - 1:+.0%}" if ctr_c else "—"
    exp["ctr"] = {"treatment": round(ctr_t, 4), "control": round(ctr_c, 4)}
try:
    exp["model_health"] = [dict(r) | {"day": str(r["day"])} for r in q(
        f"SELECT day, variant, impressions, ctr FROM `{PROJECT}.reco.model_health` ORDER BY day")]
except Exception:
    exp["model_health"] = []      # Phase 14 not run yet — not fatal

# Write once to get a size, then rewrite with the size in it — experiment.json can't
# measure a directory it is itself part of without doing this twice. It's ~2 KB; who cares.
json.dump(exp, open(OUT / "experiment.json", "w"), default=str)
kb = sum(p.stat().st_size for p in OUT.glob("*.json")) // 1024
exp["bundle_kb"] = kb
json.dump(exp, open(OUT / "experiment.json", "w"), default=str)

print(f"\n✅ bundle written to {OUT} — {kb} KB, {len(list(OUT.glob('*.json')))} files")
print("   Now go and turn everything off. The demo will still be there.")
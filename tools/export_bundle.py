import atexit
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import requests
import xgboost as xgb
from google.cloud import bigquery, storage

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from generation_contract import (  # noqa: E402
    ACTIVE_GENERATION_TABLE,
    ServingBinding,
    item_blob_name,
    manifest_blob_name,
    manifest_uri,
    parse_item_payload,
    parse_user_payload,
    sha256_bytes,
    user_blob_name,
    validate_binding,
    validate_generation_id,
    validate_manifest,
    validate_ops_image_uri,
    validate_sha256,
    validate_source_revision,
)

PROJECT = os.environ.get("PROJECT", "reco-engine-adv")
REGION = "europe-west3"
BUCKET = f"{PROJECT}-artifacts"
OUT = pathlib.Path(__file__).resolve().parent.parent / "demo" / "data"
OUT.mkdir(parents=True, exist_ok=True)
BUNDLES = OUT / "bundles"
BUNDLES.mkdir(parents=True, exist_ok=True)

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


def active_export_fence():
    rows = q(f"""SELECT generation_id, epoch, previous_generation_id, manifest_uri,
                        manifest_object_generation, manifest_sha256,
                        ops_image_uri, source_revision, retained_generation_ids,
                        champion_score, promoted_at,
                        index_name, index_endpoint, deployed_index_id
                 FROM `{PROJECT}.reco.{ACTIVE_GENERATION_TABLE}`
                 WHERE singleton = TRUE""")
    if len(rows) != 1:
        raise RuntimeError("export requires exactly one active generation")
    return dict(rows[0])


def conditional_download(blob, expected_generation=None):
    blob.reload()
    generation = int(blob.generation)
    if expected_generation is not None and generation != expected_generation:
        raise RuntimeError("immutable export input changed object generation")
    return blob.download_as_bytes(if_generation_match=generation)


export_fence = active_export_fence()
active_generation = validate_generation_id(str(export_fence["generation_id"]))
active_epoch = int(export_fence["epoch"])
if export_fence["manifest_uri"] != manifest_uri(BUCKET, active_generation):
    raise RuntimeError("active pointer does not name its exact generation manifest")
binding = validate_binding(
    export_fence["index_name"],
    export_fence["index_endpoint"],
    export_fence["deployed_index_id"],
)
active_receipt = {
    "manifest_object_generation": int(export_fence["manifest_object_generation"]),
    "manifest_sha256": validate_sha256(export_fence["manifest_sha256"]),
    "ops_image_uri": validate_ops_image_uri(export_fence["ops_image_uri"]),
    "source_revision": validate_source_revision(export_fence["source_revision"]),
}
bundle_id = (
    f"{active_generation}-{active_epoch}-{active_receipt['manifest_sha256'][:12]}-"
    f"{uuid.uuid4().hex[:8]}"
)
STAGING = pathlib.Path(tempfile.mkdtemp(prefix=f".{bundle_id}-", dir=BUNDLES))
atexit.register(shutil.rmtree, STAGING, ignore_errors=True)


def write_json(path: pathlib.Path, payload, *, default=None) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, default=default, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())


def retired_receipt(generation_id: str):
    """Return the exact most-recent retirement receipt, never an unfenced URI."""
    generation_id = validate_generation_id(generation_id)
    try:
        rows = q(f"""SELECT manifest_object_generation, manifest_sha256,
                            ops_image_uri, source_revision
                     FROM `{PROJECT}.reco.embedding_generation_retirements`
                     WHERE generation_id = '{generation_id}'
                     ORDER BY retired_epoch DESC LIMIT 1""")
    except Exception:
        return None
    if len(rows) != 1:
        return None
    row = rows[0]
    return {
        "manifest_object_generation": int(row["manifest_object_generation"]),
        "manifest_sha256": validate_sha256(row["manifest_sha256"]),
        "ops_image_uri": validate_ops_image_uri(row["ops_image_uri"]),
        "source_revision": validate_source_revision(row["source_revision"]),
    }


def load_generation(generation_id, receipt):
    if not receipt:
        raise RuntimeError("generation export requires an immutable manifest receipt")
    manifest_payload = conditional_download(
        gcs.bucket(BUCKET).blob(manifest_blob_name(generation_id)),
        receipt["manifest_object_generation"],
    )
    if sha256_bytes(manifest_payload) != receipt["manifest_sha256"]:
        raise RuntimeError("generation manifest bytes differ from the active receipt")
    manifest = validate_manifest(
        json.loads(manifest_payload),
        expected_generation_id=generation_id,
        canonical_bucket=BUCKET,
        regional_bucket=f"{PROJECT}-vectors",
        expected_binding=binding,
        expected_ops_image_uri=receipt["ops_image_uri"],
        expected_source_revision=receipt["source_revision"],
    )
    item_payload = conditional_download(
        gcs.bucket(BUCKET).blob(item_blob_name(generation_id)),
        manifest["items"]["canonical_object_generation"],
    )
    user_payload = conditional_download(
        gcs.bucket(BUCKET).blob(user_blob_name(generation_id)),
        manifest["users"]["object_generation"],
    )
    if sha256_bytes(item_payload) != manifest["items"]["sha256"]:
        raise RuntimeError("active item artifact failed its manifest digest")
    if sha256_bytes(user_payload) != manifest["users"]["sha256"]:
        raise RuntimeError("active user artifact failed its manifest digest")
    return (
        manifest,
        {row["id"]: row["embedding"] for row in parse_item_payload(
            item_payload, dimensions=manifest["dimensions"]
        )},
        {row["user_id"]: row["embedding"] for row in parse_user_payload(
            user_payload, dimensions=manifest["dimensions"]
        )},
    )


current_manifest, cur, U_cur = load_generation(active_generation, active_receipt)
DIM = current_manifest["dimensions"]
print(f"  active generation = {active_generation}@{active_epoch}, dim = {DIM}")
previous_generation = export_fence.get("previous_generation_id")
if previous_generation:
    previous_receipt = retired_receipt(str(previous_generation))
    if previous_receipt:
        previous_manifest, prev, U_prev = load_generation(
            str(previous_generation), previous_receipt
        )
        if previous_manifest["dimensions"] != DIM:
            raise RuntimeError("active and previous generations use different dimensions")
    else:
        print("  ! previous generation has no exact retirement receipt; comparison disabled")
        previous_generation = None
        prev = U_prev = None
else:
    prev = U_prev = None
print(
    f"  previous generation = {previous_generation}"
    if previous_generation
    else "  no previous retained generation — retrain demo will be disabled"
)

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

# ── 3. User vectors from the same immutable active/previous manifests ──
print("· user vectors + features + ground truth")

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
    raise RuntimeError("previous item/user generation is not a matched immutable pair")

item_payload = {
    "dim": DIM, "categories": cats, "ids": item_ids,
    "category": [r["category"] for r in items],
    "price": [float(r["price"]) for r in items],
    "popularity_7d": [int(r["popularity_7d"]) for r in items],
    "emb": flat(I1), "xy": flat(project2(I1)),
}
if I2 is not None:
    item_payload |= {"emb_v2": flat(I2), "xy_v2": flat(project2(I2))}

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

# Re-read the pointer fence after every immutable generation input is loaded.
# Temporary writes are private to this export and cannot become browser-visible.
if active_export_fence() != export_fence:
    raise RuntimeError("active generation changed during bundle export; retry")
write_json(STAGING / "items.json", item_payload)
write_json(STAGING / "users.json", user_payload)

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

# Ship exactly what the browser needs to reproduce this (label-index + per-group softmax).
# This is intentionally a different, explicit schema path from mock_bundle.py's one-hot binary
# classifier. Optional-field inference made the browser accept both while scoring one of them
# with the other's objective.
write_json(STAGING / "ranker.json", {
    "format": "reco-ranker",
    "schema_version": 1,
    "features": FEATS,
    "encoding": {
        "type": "label_index",
        "source_feature": "category",
        "feature": CAT,
        "feature_index": cat_idx,
        "vocabulary": vocab,
        "unknown": "missing",
    },
    "objective": {
        "type": "multi:softprob",
        "groups": n_groups,
        "positive_group": pos_group,
        "tree_groups": tree_info,
    },
    "trees": trees,
})
print(f"  ✓ staged ranker.json — {len(trees)} trees, {n_groups} class groups, residual {worst:.1e}")
print("· golden set from the live API")
api = subprocess.run(["gcloud", "run", "services", "describe", "reco-api",
                      "--region", REGION, "--format=value(status.url)"],
                     capture_output=True, text=True).stdout.strip()
golden_api = []
api_token = os.environ.get("API_BEARER_TOKEN")
if api and api_token:
    headers = {"Authorization": f"Bearer {api_token}"}
    identity = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if identity:
        headers["X-Serverless-Authorization"] = f"Bearer {identity}"
    for u in user_ids[:25]:
        try:
            r = requests.post(
                f"{api}/recommend",
                json={"user_id": u, "request_id": str(uuid.uuid4())},
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            body = r.json()
            if body.get("served_by") == "personalized":
                golden_api.append({"user_id": u, "items": body["items"]})
        except Exception as e:
            print("   !", u, repr(e))
    print(f"  {len(golden_api)} personalised responses recorded")
else:
    reason = "reco-api not found" if not api else "API_BEARER_TOKEN not set"
    print(f"  ! {reason} — no live API golden rows will be exported")
# ranker golden — 20 rows the browser must reproduce, with BQML's own prob alongside for the fidelity panel
golden_ranker = [{"row": {f: (str(dict(r)[f]) if f == CAT else float(dict(r)[f])) for f in FEATS},
                  "browser": score(encode(dict(r))), "ml_predict": float(r["prob"])}
                 for r in gold[:20]]
write_json(STAGING / "golden.json", {"api": golden_api, "ranker": golden_ranker})

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

# Write once to get a size, then rewrite with the size in it. Both writes remain
# in the private staging directory until the active receipt is fenced again.
write_json(STAGING / "experiment.json", exp, default=str)
kb = sum(p.stat().st_size for p in STAGING.glob("*.json")) // 1024
exp["bundle_kb"] = kb
write_json(STAGING / "experiment.json", exp, default=str)

# No generated file becomes visible before all cloud reads finish and the exact
# active generation, manifest object generation/hash, provenance, score, and
# Vertex binding are rechecked. Re-reading the immutable manifest also catches
# a pointer row that somehow retained its fields while object content drifted.
if active_export_fence() != export_fence:
    raise RuntimeError("active generation changed before atomic bundle publication")
load_generation(active_generation, active_receipt)
if active_export_fence() != export_fence:
    raise RuntimeError("active generation changed during final receipt verification")

for directory in (STAGING, BUNDLES):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

FINAL_BUNDLE = BUNDLES / bundle_id
if FINAL_BUNDLE.exists():
    raise RuntimeError(f"refusing to replace existing versioned bundle {FINAL_BUNDLE}")
os.replace(STAGING, FINAL_BUNDLE)
bundle_descriptor = os.open(BUNDLES, os.O_RDONLY)
try:
    os.fsync(bundle_descriptor)
finally:
    os.close(bundle_descriptor)

pointer = {
    "schema_version": 1,
    "bundle": f"bundles/{bundle_id}",
    "active_generation": active_generation,
    "active_epoch": active_epoch,
    "manifest_object_generation": active_receipt["manifest_object_generation"],
    "manifest_sha256": active_receipt["manifest_sha256"],
}
pointer_tmp = OUT / f".current-{uuid.uuid4().hex}.json"
write_json(pointer_tmp, pointer)
os.replace(pointer_tmp, OUT / "current.json")
out_descriptor = os.open(OUT, os.O_RDONLY)
try:
    os.fsync(out_descriptor)
finally:
    os.close(out_descriptor)

print(f"\n✅ atomic bundle published at {FINAL_BUNDLE} — {kb} KB, {len(list(FINAL_BUNDLE.glob('*.json')))} files")
print("   Now go and turn everything off. The demo will still be there.")

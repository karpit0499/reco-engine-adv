"""diag_ranker.py — read-only diagnosis of the ranker fidelity gap. Writes nothing.
    PROJECT=reco-engine-adv python tools/diag_ranker.py
"""
import json, math, os, pathlib, subprocess, tempfile
import numpy as np
import xgboost as xgb
from google.cloud import bigquery, storage

PROJECT = os.environ.get("PROJECT", "reco-engine-adv")
BUCKET = f"{PROJECT}-artifacts"
bq = bigquery.Client(project=PROJECT)
gcs = storage.Client(project=PROJECT)
q = lambda sql: list(bq.query(sql).result())
f32 = np.float32

tmp = pathlib.Path(tempfile.mkdtemp())
subprocess.run(["bq", "extract", "--destination_format=ML_XGBOOST_BOOSTER",
                "-m", f"{PROJECT}:reco.ranker", f"gs://{BUCKET}/export/ranker"], check=True)
bkt = gcs.bucket(BUCKET)
_sc = lambda n: bkt.blob(f"export/ranker/{n}").download_as_text()
bkt.blob("export/ranker/model.bst").download_to_filename(tmp / "model.bst")
meta = json.loads(_sc("assets/model_metadata.json"))

b = xgb.Booster(); b.load_model(str(tmp / "model.bst"))
b.save_model(str(tmp / "r.json"))
raw = json.load(open(tmp / "r.json"))
bs = float(str(raw["learner"]["learner_model_param"]["base_score"]).strip("[]"))
base_margin = math.log(bs / (1 - bs))
trees = [{"l": t["left_children"], "r": t["right_children"], "i": t["split_indices"],
          "c": [float(v) for v in t["split_conditions"]], "d": t["default_left"]}
         for t in raw["learner"]["gradient_booster"]["model"]["trees"]]

print("class_names       :", meta.get("class_names"))
print("base_score raw    :", raw["learner"]["learner_model_param"]["base_score"], "-> bs", bs)
print("base_margin       :", base_margin)
print("num trees         :", len(trees))
internal = sum(1 for t in trees for lc in t["l"] if lc != -1)
print("total split nodes :", internal, "(0 would mean the trees carry no signal)")

# fixed layout: null_filler dropped, one-hot inline (11 cols). Layout is irrelevant to this
# diagnosis — every layout gave the same residual — so any consistent one is fine here.
ohe = next(f for f in meta["feature_names"]
           if meta["features"].get(f, {}).get("encode_type") == "ohe")
oi = meta["feature_names"].index(ohe)
vocab = [v for v in _sc(f"assets/{oi}_categorical_label.txt").splitlines() if v and v != "_null_filler"]
FEATS = []
for f in meta["feature_names"]:
    FEATS += [f"{ohe}={v}" for v in vocab] if f == ohe else [f]
cat_names = {v: f"{ohe}={v}" for v in vocab}

def walk(t, row):
    i = 0
    while t["l"][i] != -1:
        v = row.get(FEATS[t["i"][i]])
        go = bool(t["d"][i]) if v is None else (f32(v) < f32(t["c"][i]))
        i = t["l"][i] if go else t["r"][i]
    return t["c"][i]

gold = q(f"""
  SELECT events_7d, purchases_30d, distinct_items_7d, category, price, popularity_7d,
         same_category,
         (SELECT p.prob FROM UNNEST(predicted_label_probs) p WHERE p.label = 1) AS prob
  FROM ML.PREDICT(MODEL `{PROJECT}.reco.ranker`,
                  (SELECT * EXCEPT(label) FROM `{PROJECT}.reco.rank_training` LIMIT 200))
""")

print("\n row |   prob   |  score   | 1-score  | sum_trees | logit")
print(" ----+----------+----------+----------+-----------+--------")
d_score, d_flip = 0.0, 0.0
for k, r in enumerate(gold):
    row = {f: 0 for f in FEATS}
    row |= {"events_7d": r["events_7d"], "purchases_30d": r["purchases_30d"],
            "distinct_items_7d": r["distinct_items_7d"], "price": float(r["price"]),
            "popularity_7d": r["popularity_7d"], "same_category": r["same_category"]}
    row[cat_names[r["category"]]] = 1
    st = sum(walk(t, row) for t in trees)
    logit = base_margin + st
    sc = 1 / (1 + math.exp(-logit))
    prob = float(r["prob"])
    d_score = max(d_score, abs(sc - prob))
    d_flip = max(d_flip, abs((1 - sc) - prob))
    if k < 8:
        print(f" {k:3d} | {prob:.5f} | {sc:.5f} | {1-sc:.5f} | {st:+.5f} | {logit:+.3f}")

print(f"\nmax |score      - prob| over 200 = {d_score:.3e}")
print(f"max |(1 - score) - prob| over 200 = {d_flip:.3e}")
print("\nread-out:")
print("  if the (1-score) column matches prob  -> class orientation is flipped (class_names order)")
print("  if score is ~constant across rows     -> base_margin saturating / trees not read")
print("  if score-prob is a constant offset    -> base_margin value is wrong")
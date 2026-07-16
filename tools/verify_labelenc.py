# tools/verify_labelenc.py (v2) — correct encoding (label-encode category) AND correct
# scoring. BQML BOOSTED_TREE_CLASSIFIER is multi:softprob: one tree per class per round.
# Sum trees by class group (tree_info) then softmax — NOT sum-all-trees then sigmoid.
import os, json, math, pathlib, tempfile, subprocess
import numpy as np, xgboost as xgb
from google.cloud import bigquery, storage

PROJECT = os.environ.get("PROJECT", "reco-engine-adv")
BUCKET  = os.environ.get("BUCKET",  f"{PROJECT}-artifacts")
f32 = np.float32

tmp = pathlib.Path(tempfile.mkdtemp())
subprocess.run(["bq","extract","--destination_format=ML_XGBOOST_BOOSTER",
                "-m", f"{PROJECT}:reco.ranker", f"gs://{BUCKET}/export/ranker"], check=True)
bkt = storage.Client(project=PROJECT).bucket(BUCKET)
bkt.blob("export/ranker/model.bst").download_to_filename(str(tmp/"model.bst"))
meta = json.loads(bkt.blob("export/ranker/assets/model_metadata.json").download_as_text())

booster = xgb.Booster(); booster.load_model(str(tmp/"model.bst"))
booster.save_model(str(tmp/"ranker.json"))
raw = json.load(open(tmp/"ranker.json")); learner = raw["learner"]
gbm = learner["gradient_booster"]["model"]
objective = learner["objective"]
num_class = int(learner["learner_model_param"].get("num_class", "0") or "0")
tree_info = gbm.get("tree_info", [0]*len(gbm["trees"]))
n_groups  = max(num_class, max(tree_info)+1, 1)
print("objective:", objective, "| num_class:", num_class,
      "| tree groups:", n_groups, "| n trees:", len(gbm["trees"]))

bs = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
base_margin = math.log(bs/(1-bs)) if bs not in (0.0, 1.0) else 0.0
trees = [{"l":t["left_children"],"r":t["right_children"],"i":t["split_indices"],
          "c":[float(v) for v in t["split_conditions"]],"d":t["default_left"]}
         for t in gbm["trees"]]

FEATS = meta["feature_names"]
CAT = [f for f in FEATS if meta["features"].get(f,{}).get("encode_type")=="ohe"][0]
cat_idx = FEATS.index(CAT)
vocab = bkt.blob(f"export/ranker/assets/{cat_idx}_categorical_label.txt").download_as_text().splitlines()
class_names = [str(c) for c in meta.get("class_names", ["0","1"])]
print("features:", FEATS, "| categorical:", CAT, "| classes:", class_names)

def encode(r):
    out=[]
    for f in FEATS:
        if f==CAT:
            v=str(r[f]); out.append(float(vocab.index(v)) if v in vocab else None)
        else:
            out.append(float(r[f]))
    return out

def walk(t,row):
    i=0
    while t["l"][i]!=-1:
        x=row[t["i"][i]]
        go=(t["d"][i]==1) if x is None else (f32(x)<f32(t["c"][i]))
        i=t["l"][i] if go else t["r"][i]
    return t["c"][i]

def score(row):
    if n_groups>=2:                         # multi:softprob — per-class sums, softmax
        m=[0.0]*n_groups
        for k,t in enumerate(trees):
            m[tree_info[k]] += walk(t,row)
        mx=max(m); e=[math.exp(v-mx) for v in m]; s=sum(e)
        return (e[class_names.index("1")]/s)
    return 1/(1+math.exp(-(base_margin+sum(walk(t,row) for t in trees))))   # binary fallback

bq = bigquery.Client(project=PROJECT)
gold = list(bq.query(f"""
  SELECT {', '.join(FEATS)},
         (SELECT p.prob FROM UNNEST(predicted_label_probs) p WHERE p.label = 1) AS prob
  FROM ML.PREDICT(MODEL `{PROJECT}.reco.ranker`,
                  (SELECT * EXCEPT(label) FROM `{PROJECT}.reco.rank_training` LIMIT 200))
""").result())

worst=max(abs(score(encode(dict(r)))-float(r["prob"])) for r in gold)
print(f"\nmax |hand-walk - ML.PREDICT| over 200 rows = {worst:.2e}")
print("PASS — encoding + softmax reproduce BQML. Safe to edit export_bundle.py."
      if worst<1e-4 else "STILL OFF — paste the objective/num_class line and this residual.")
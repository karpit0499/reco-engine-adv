# tools/diag_leaf.py — reproduces export_bundle.py's exact model load, then reports
# where BQML's booster actually stores leaf values. Run: python tools/diag_leaf.py
import os, json, pathlib, tempfile, subprocess
import xgboost as xgb
from google.cloud import storage

PROJECT = os.environ.get("PROJECT", "reco-engine-adv")
BUCKET  = os.environ.get("BUCKET",  f"{PROJECT}-artifacts")
print(f"project={PROJECT}  bucket={BUCKET}  xgboost={xgb.__version__}")

# --- same three steps as export_bundle.py lines ~175-187 ---
tmp = pathlib.Path(tempfile.mkdtemp())
subprocess.run(["bq", "extract", "--destination_format=ML_XGBOOST_BOOSTER",
                "-m", f"{PROJECT}:reco.ranker", f"gs://{BUCKET}/export/ranker"], check=True)
storage.Client(project=PROJECT).bucket(BUCKET) \
    .blob("export/ranker/model.bst").download_to_filename(str(tmp / "model.bst"))
booster = xgb.Booster(); booster.load_model(str(tmp / "model.bst"))

# (1) xgboost's OWN parse. For a Leaf row the value sits in the 'Gain' column. Ground truth.
df = booster.trees_to_dataframe(); leaves = df[df.Feature == "Leaf"]
print("\n[trees_to_dataframe] leaf values: min=%.5f max=%.5f  nonzero=%d/%d"
      % (leaves.Gain.min(), leaves.Gain.max(), int((leaves.Gain != 0).sum()), len(leaves)))
print(leaves[["Tree", "Node", "Gain"]].head(6).to_string(index=False))

# (2) the raw arrays your walk reads — is the value in split_conditions or base_weights?
booster.save_model(str(tmp / "ranker.json"))
raw = json.load(open(tmp / "ranker.json"))
t0 = raw["learner"]["gradient_booster"]["model"]["trees"][0]
leaf_idx = [k for k, lc in enumerate(t0["left_children"]) if lc == -1]
print("\n[raw arrays] tree 0 leaves — node : split_conditions : base_weights")
for k in leaf_idx[:6]:
    print(f"  {k:>4} : {float(t0['split_conditions'][k]):>14.6f} : {float(t0['base_weights'][k]):>12.6f}")
print("  split_conditions all-zero at leaves?", all(float(t0['split_conditions'][k]) == 0 for k in leaf_idx))
print("  base_weights     all-zero at leaves?", all(float(t0['base_weights'][k]) == 0 for k in leaf_idx))

bs = float(str(raw["learner"]["learner_model_param"]["base_score"]).strip("[]"))
print(f"\n  base_score={bs:.6f}  -> collapse prob if leaves read 0 = {bs:.4f}")
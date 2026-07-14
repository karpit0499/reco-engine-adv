import json
import sys

from google.cloud import aiplatform, storage

PROJECT, REGION = "reco-engine-adv", "europe-west3"
# The EU (artifacts) bucket, deliberately — NOT the europe-west3 vectors mirror.
# This script downloads the vectors itself and pushes them through the API with
# upsert_datapoints(); no GCS location constraint applies. The regional mirror only
# exists for contents_delta_uri, which is read at index CREATION and never again.
BUCKET, BLOB = "reco-engine-adv-artifacts", "embeddings/items.json"
INDEX = sys.argv[1]      # full index resource name

aiplatform.init(project=PROJECT, location=REGION)
index = aiplatform.MatchingEngineIndex(index_name=INDEX)

blob = storage.Client(project=PROJECT).bucket(BUCKET).blob(BLOB)
dps = []
for line in blob.download_as_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    dps.append({"datapoint_id": r["id"], "feature_vector": r["embedding"]})

if not dps:
    sys.exit(f"gs://{BUCKET}/{BLOB} is empty — run training/train.py first")

for i in range(0, len(dps), 100):
    index.upsert_datapoints(datapoints=dps[i:i + 100])
    print(f"  upserted {min(i + 100, len(dps))}/{len(dps)}")
print(f"done — {len(dps)} datapoints refreshed in the live index")
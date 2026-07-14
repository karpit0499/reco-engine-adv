import sys

from google.cloud import aiplatform, bigquery

PROJECT, REGION = "reco-engine-adv", "europe-west3"
INDEX_ENDPOINT = sys.argv[1]        # full resource name — see the shell snippet below
DEPLOYED_INDEX_ID = "items_v1"
TOP_K = 10

aiplatform.init(project=PROJECT, location=REGION)
bq = bigquery.Client(project=PROJECT)
ep = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT)

catalog = {r["item_id"]: r["category"] for r in bq.query(
    f"SELECT item_id, category FROM `{PROJECT}.reco.catalog`").result()}
truth = {r["user_id"]: r["preferred_category"] for r in bq.query(
    f"SELECT user_id, preferred_category FROM `{PROJECT}.reco.users_truth`").result()}
uvecs = {r["user_id"]: list(r["embedding"]) for r in bq.query(
    f"SELECT user_id, embedding FROM `{PROJECT}.reco.user_embeddings_latest`").result()}

if not uvecs:
    sys.exit("no user embeddings — run training/train.py first (Phase 7)")

sample = [u for u in list(uvecs) if u in truth][:50]
hits = total = 0
for n, uid in enumerate(sample):
    res = ep.find_neighbors(deployed_index_id=DEPLOYED_INDEX_ID,
                            queries=[uvecs[uid]], num_neighbors=TOP_K)
    cats = [catalog.get(m.id, "?") for m in res[0]]
    hits += sum(c == truth[uid] for c in cats)
    total += TOP_K
    if n < 5:
        print(f"{uid} prefers {truth[uid]:<12} top-{TOP_K} categories: {cats}")

print(f"\npreferred-category share in top-{TOP_K}: {hits / total:.1%} "
      f"(random baseline = 20% with 5 categories)")
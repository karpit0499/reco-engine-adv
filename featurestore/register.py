import vertexai
# The Feature Store SDK lives under vertexai preview modules and the exact
# import path has moved more than once. If this import fails, open the current
# "Vertex AI Feature Store" Python samples and copy the import line — the
# CONCEPTS (feature group → BigQuery source + entity-id columns → features)
# are stable even when the classes get reshuffled.
from vertexai.resources.preview import feature_store as fs

PROJECT, REGION = "reco-engine-adv", "europe-west3"
vertexai.init(project=PROJECT, location=REGION)

# NOTE: the source is the APPEND-ONLY table, not the _latest view. The registry
# is designed around feature history — entity_id_columns + a feature_timestamp
# column is exactly the shape it expects, and it resolves "current value" itself.
user_fg = fs.FeatureGroup.create(
    name="user_features",
    source=fs.utils.FeatureGroupBigQuerySource(
        uri=f"bq://{PROJECT}.reco.user_features",
        entity_id_columns=["user_id"],
    ),
)
for col in ["events_7d", "purchases_30d", "distinct_items_7d", "top_category"]:
    user_fg.create_feature(name=col)

item_fg = fs.FeatureGroup.create(
    name="item_features",
    source=fs.utils.FeatureGroupBigQuerySource(
        uri=f"bq://{PROJECT}.reco.item_features",
        entity_id_columns=["item_id"],
    ),
)
for col in ["category", "price", "popularity_7d"]:
    item_fg.create_feature(name=col)

print("registered 2 feature groups + 7 features")
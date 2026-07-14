from google.cloud import aiplatform

PROJECT, REGION = "reco-engine-adv", "europe-west3"
aiplatform.init(project=PROJECT, location=REGION)

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="item-index",
    contents_delta_uri="gs://reco-engine-adv-vectors/embeddings",   # ← CHANGED
    dimensions=32,
    approximate_neighbors_count=50,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="STREAM_UPDATE",
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=100,
)
print("index:", index.resource_name)

endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="item-index-endpoint",
    public_endpoint_enabled=True,
)
print("endpoint:", endpoint.resource_name)

endpoint.deploy_index(index=index, deployed_index_id="items_v1")
print("deployed as items_v1")
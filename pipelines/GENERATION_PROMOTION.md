# Generation-filtered embedding promotion

Retraining is a fenced compensating saga across GCS, BigQuery, and Vertex
Vector Search. It is not a cross-service atomic transaction. Its safety
property is narrower and testable: a request names exactly one active
generation, and a failed or held candidate cannot become visible through that
generation filter.

## Immutable candidate contract

Every run receives a fresh `generation_id`. Training writes these objects with
GCS create-if-absent preconditions:

```text
gs://<project>-artifacts/candidate-generations/<generation_id>/items.json
gs://<project>-vectors/candidate-generations/<generation_id>/items.json
gs://<project>-artifacts/candidate-generations/<generation_id>/users.json
gs://<project>-artifacts/candidate-generations/<generation_id>/manifest.json
```

The manifest is uploaded last. It records both item-object generations, the
user-object generation, canonical SHA-256 digests, counts, vector dimension,
item and user canaries, generation-specific model paths, and the exact Vertex
index resource, index endpoint resource, and deployed-index ID. User vectors
are also staged in `reco.user_embedding_generations` under the same generation
key; training never changes the live user view.

Evaluation reloads object metadata before every conditional download, checks
both item copies and the immutable user object, reconstructs the canonical user
hash from the generation-keyed BigQuery rows, and evaluates that matched
user/item pair. An empty active-state table is handled as a first generation.

## Promotion sequence

For an existing active generation, promotion performs these steps:

1. Compare the evaluated active generation, epoch, and exact Vertex binding to
   the current state.
2. Acquire a lease containing a unique owner token, monotonically increasing
   fence epoch, phase, acquisition/heartbeat timestamps, expiry, and pending
   generation/manifest. A non-stale concurrent lease fails closed. Stale
   recovery uses a compare-and-set transaction and revalidates every fence.
3. Upsert candidate items alongside older items. Vertex datapoint IDs are
   `<generation_id>::<item_id>` and every datapoint has a categorical
   `generation=<generation_id>` restrict. Promotion does not delete old IDs.
4. Through the exact configured deployed endpoint, call
   `read_index_datapoints` until every candidate ID, vector, and restrict is
   visible. Then issue a generation-restricted self-query for the manifest
   canary and reject any mixed-generation result.
5. Reload and conditionally read the immutable artifacts again, then recreate
   and verify the canonical BigQuery user hash immediately before finalization.
6. In one BigQuery transaction, append accepted users to promoted history and
   switch the active generation, live-user view input, epoch, manifest, and
   champion score. The lease is cleared while its last fence epoch is retained.

The API reads the active pointer and matching user vector in one BigQuery
query, sends the exact active generation as a Vertex `Namespace` filter, and
strips only the matching namespaced prefix. If promotion occurs while a request
is in flight, that request continues querying its captured old generation;
old and new vectors are never combined.

A held evaluation never invokes promotion. If upsert, propagation, verification,
or finalization fails, the active pointer, active user generation, and champion
remain unchanged. Candidate datapoints may remain staged but are invisible to
active-generation queries. The immediately previous generation remains in
Vertex for rollback and in-flight requests.

## First generation and cleanup

First-generation seeding uses a separate GCS create-if-absent initialization
lock. Its owner, monotonic epoch, binding, heartbeat, phase, and expiry are
conditionally updated and rechecked before the idempotent BigQuery singleton
`MERGE`. This lock, rather than a BigQuery uniqueness assertion, serializes
bootstrap. A stale initializer can be recovered only after the active state and
lock object generation are revalidated.

Vertex generations are never deleted by promotion. Cleanup is explicit,
dry-run by default, requires the observed active generation and epoch, acquires
the same fenced lease, refuses the active and immediately previous generations,
and defaults to seven days of retention:

```bash
python featurestore/gc_generation.py \
  --generation-id <retired-generation> \
  --expected-active-generation <active-generation> \
  --expected-active-epoch <active-epoch> \
  --index-name projects/<p>/locations/<r>/indexes/<id> \
  --index-endpoint projects/<p>/locations/<r>/indexEndpoints/<id> \
  --deployed-index-id <deployed-id>
```

Review the dry run, then add `--execute`. Cleanup removes only namespaced Vertex
datapoints. It retains immutable GCS objects and BigQuery audit data.

`tools/export_bundle.py` also reads the active generation and epoch once, loads
the exact immutable item/user pair from its manifest, and rechecks the pointer
fence before writing the bundle. It never combines a mutable alias with an
independently selected history row.

## Offline verification

The pipeline image pins `google-cloud-aiplatform==1.165.1`. Regenerate the KFP
v2 specification and run focused tests without cloud credentials:

```bash
python pipelines/reco_pipeline.py
python -m unittest -v pipelines.tests.test_generation_promotion
python -m unittest -v api.tests.test_contracts api.tests.test_main
```

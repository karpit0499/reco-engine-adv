# Vertex Vector Search runtime evidence

This directory contains a read/query-only, fail-closed harness for one active
generation of an existing Vertex AI Vector Search deployment. It does not
create, deploy, update, promote, garbage-collect, or delete a cloud resource.
The default command is a **no-cloud dry run**. Private inputs and all generated
reports are ignored by Git.

The harness reports two distinct diagnostics:

- client-observed `find_neighbors` wall time for every warmup and measured call,
  with measured-success p50/p95; and
- top-K set overlap between the live ANN response and an exact scan of the
  complete active item-vector catalog.

Set overlap is an ANN/index diagnostic. It is not Recall@K, NDCG, relevance,
business lift, or recommendation quality. The sealed MovieLens evaluation is a
separate protocol and is never opened by this command.

## Evidence boundary

The private input must match [`input.schema.json`](input.schema.json). It binds
the vectors to all of the following:

1. the exact stable row expected in
   `<project>.reco.active_embedding_generation`, including generation, epoch,
   fence epoch, champion, retained generations, serving target, immutable
   manifest receipt, build image digest, and source revision;
2. the immutable promotion manifest URI, GCS object generation, and SHA-256;
3. the manifest's canonical item count, canonical JSONL SHA-256, item-object
   generation, and embedding dimension; and
4. the exact Vertex index, endpoint, deployed-index ID, configured distance
   metric, and `STREAM_UPDATE` mode.

The local catalog is canonicalized with the same production helper used by
promotion. Its row count and SHA-256 must exactly reproduce the manifest
commitment. A sampled or changed catalog is refused, because its exact-scan
comparison would not describe the same population as the ANN index.

Immediately before and after querying, the harness independently:

- reads the singleton BigQuery active pointer and refuses an in-progress lease;
- generation-pins and hashes the manifest download;
- strictly checks the current promotion-manifest contract; and
- fetches live Index and IndexEndpoint metadata, validating dimension,
  distance metric, update method, and deployed mapping.

A pass requires both snapshots to have identical stable identities. A pointer,
manifest, index configuration, or deployed mapping change fails the run. Every
ANN request uses the active generation filter and namespaced datapoint contract:

```text
Namespace("generation", [<active-generation>], [])
datapoint ID: <active-generation>::<raw-item-id>
```

Missing or mixed-generation responses, duplicate items, items outside the
complete manifest-bound catalog, warmup failures, measured failures, and
postflight read failures are recorded and fail closed. Exact results are never
substituted for a failed ANN call.

Source provenance and the runner's client region cannot be independently
verified by this process. Both appear in reports explicitly as
`operator_attested`. Query identifiers, embeddings, and returned item IDs are
omitted from reports.

## Dry-run plan first

Use Python 3.12. Create the private input outside the repository, or under the
ignored `benchmarks/vector_search_runtime/private/` directory. Install the
pinned clients, then run the command **without** `--execute-live`:

```bash
python -m pip install -r benchmarks/vector_search_runtime/requirements.txt

python -m benchmarks.vector_search_runtime.runtime_benchmark \
  --artifact /secure/path/active-generation.runtime-input.json \
  --output benchmarks/vector_search_runtime/results/plan.json \
  --active-pointer-table PROJECT.reco.active_embedding_generation \
  --index-resource projects/PROJECT/locations/REGION/indexes/INDEX \
  --index-endpoint projects/PROJECT/locations/REGION/indexEndpoints/ENDPOINT \
  --deployed-index-id DEPLOYED_ID \
  --generation-id ACTIVE_GENERATION \
  --promotion-epoch ACTIVE_EPOCH \
  --client-region RUNNER_REGION \
  --top-k 10 \
  --warmups 5
```

Dry-run mode validates the private input and all hard caps, computes the exact
planned call inventory, writes an atomic diagnostic matching
[`diagnostic.schema.json`](diagnostic.schema.json), prints the required dynamic
confirmation phrase, and makes **zero cloud calls**.

The hard limits are 100,000 catalog items, 500 measured queries, 50 warmups,
top-K 100, 500 million exact scalar operations, a 256 MiB input, and 561
potentially billable API calls. The plan enumerates two BigQuery pointer
queries, two GCS metadata reads, two generation-pinned GCS downloads, two Index
metadata reads, three IndexEndpoint metadata reads (including query-client
bootstrap), and the exact number of
`find_neighbors` calls.

## Execute the reviewed plan

Live execution requires Application Default Credentials that can read the
pointer table and manifest and read/query the stated Vertex resources. It also
requires a clean Git checkout. Repeat the dry-run command with these additions,
copying the phrase **exactly** from that dry-run output:

```text
--execute-live
--attest-source-is-production
--confirmation "EXECUTE <N> POTENTIALLY BILLED CLOUD API CALLS FOR <generation>@<epoch>"
```

The production-source flag is an operator attestation, not independent
verification. Any mismatched phrase is refused before repository inspection,
credentials, or cloud access.

Completed pass/fail results match
[`result.schema.json`](result.schema.json). Reports are written with mode 0600
using fsync and an atomic replace. Once arguments have parsed, local validation,
authorization, credential, and preflight refusals also produce an atomic,
schema-valid diagnostic when the destination is writable. Error reports include
only a stage and exception type, not raw service responses or identifiers.

## Validation without cloud access

```bash
python -m unittest discover -s benchmarks/vector_search_runtime/tests -v
```

The tests inject BigQuery, GCS, Index, IndexEndpoint, and query clients. They
generate schema-valid pass and failure reports, enforce cross-field invariants,
exercise dry-run/refusal behavior, and verify immutable manifest reads, live
serving checks, catalog completeness, billing caps, two-snapshot fencing,
privacy omission, and atomic output. They never obtain credentials or call GCP.

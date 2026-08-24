# Technical documentation

This document describes the audited repository state. It distinguishes checked
offline evidence from cloud behavior that requires an explicitly authorized run.

## 1. Evidence model

| Evidence | Data | What it supports | What it does not support |
|---|---|---|---|
| real-data benchmark | pinned MovieLens 1M | temporal offline recommendation comparison | production/shop generalization or endpoint latency |
| synthetic fixture | generated users/items/preferences | dataflow, scorer, experiment, and artifact wiring | real recommendation quality or business lift |
| A/B simulations | seeded Bernoulli/user-assignment designs | FPR, coverage, power, SRM, stopping implementation | observed human behavior |
| endpoint harness | operator-supplied real embeddings and existing Vertex target | raw ANN latency and exact-vs-ANN overlap after a run | recommendation quality |

No number crosses these boundaries in the README or browser demo.

## 2. Offline recommendation protocol

The authoritative files live in `benchmarks/recommender/`.

- Source: official MovieLens 1M archive pinned by SHA-256; raw data is ignored.
- Split: global chronological timestamp blocks. Equal timestamps are never split.
- Fitting: validation uses events strictly before its cutoff; the test artifact is
  physically separate and inaccessible to `validate.py`.
- Relevance: rating 4 or 5; candidate masking excludes every earlier rating,
  including ratings below 4.
- Population: predeclared warm users plus users with no prior event. Sparse-history
  users outside the rule are excluded. Cold users receive exact popularity fallback.
- Candidates: the complete 3,883-item catalog for all models; previously seen items
  are removed and cold catalog items remain eligible.
- Models: popularity, cosine item-kNN, ALS, BPR, two-tower ID embeddings, and
  ALS-retrieve-200 plus a logistic reranker.
- Metrics: Recall/NDCG/MRR/user coverage at 10 and 20 with 2,000 paired user-bootstrap
  resamples. Catalog coverage is a non-additive full-cohort descriptive statistic.
- Cohorts: warm/cold users and warm/cold target items are reported separately.

Each family has one fixed representative configuration. Validation selects the maximum
NDCG@10 with a declared tie order; it does not optimize every family.

### 2.1 Seal and one-time test

The selection receipt binds protocol, config, split, validation evidence, complete
benchmark Python source, and pinned requirements. Test authorization recomputes the
selection and strongest simple baseline.

The fixed test ledger is created mode `0600` with `O_CREAT|O_EXCL`, followed by file and
directory `fsync`, before the test artifact is read or hashed. It contains a unique
owner and nonce. All transitions require those tokens; the fixed result path is also
exclusive. Concurrent attempts produce one winner, and mismatch/crash/failure consumes
the attempt.

The test compares the selected model with the frozen strongest simple baseline using
the same users and paired bootstrap. The acceptance rule is a strictly positive lower
95% confidence bound for the NDCG@10 difference. A failed rule is still final evidence
and cannot trigger test-driven tuning.

The authorized ordinal-1 test run from frozen commit `e67ae1ce` completed. ALS
NDCG@10 was `0.04540`; popularity was `0.05398`. Their paired difference was
`-0.00858` with 95% CI `[-0.02075, +0.00320]`, so the acceptance gate failed. The
completed result and access ledger are the authoritative post-test evidence.

## 3. Immutable embedding generation

Training produces four create-once objects per `generation_id`:

```text
candidate-generations/<generation_id>/items.json   (canonical bucket)
candidate-generations/<generation_id>/items.json   (regional Vector bucket)
candidate-generations/<generation_id>/users.json
candidate-generations/<generation_id>/manifest.json
```

Item and user payloads are canonical sorted JSONL with unique IDs and finite,
fixed-dimension vectors. The manifest records exact GCS object generations, SHA-256
digests, counts, dimensions, canaries, model paths, and a fully qualified binding:

- Vertex index resource;
- Vertex index-endpoint resource;
- deployed-index ID;
- categorical restrict namespace `generation`.

Training mirrors the canonical item payload to the regional bucket and verifies both
copies. User vectors are staged in a generation-keyed BigQuery table; training never
changes the active user view or champion.

## 4. Evaluation and promotion

Evaluation conditionally downloads the exact manifest-bound objects, validates the two
item copies, reconstructs the canonical user hash from both GCS and staged BigQuery
rows, and scores the matched generation. The gate is side-effect free.

Promotion is a compensating workflow with request-level atomicity through generation
filtering:

1. revalidate the evaluated active generation, epoch, and serving binding;
2. acquire a BigQuery lease with owner, monotonically increasing fence epoch, phase,
   heartbeat, and expiry;
3. upsert datapoints as `<generation_id>::<item_id>` with
   `generation=<generation_id>` restrict, retaining earlier generations;
4. poll `read_index_datapoints` through the exact deployed endpoint until every ID,
   vector, and restrict is visible, then run a restricted self-query on the manifest
   canary and reject mixed-generation results;
5. reload/rehash candidate items and users under the current fence;
6. transactionally append promoted users and switch active generation, live-user view,
   epoch, manifest, champion score, and cleared lease.

A held/failed candidate cannot move the active pointer. Staged candidate datapoints may
remain, but active queries cannot see them. First-generation seeding uses a separate
GCS create-if-absent lock with owner/epoch/heartbeat/expiry and a fenced idempotent
singleton `MERGE`; BigQuery uniqueness assumptions are not used as a mutex.

Cleanup is a separate dry-run-by-default command. It acquires the same fence, requires
the observed active generation/epoch, enforces a retention age, and refuses active or
immediately prior generations. It deletes only namespaced Vertex datapoints and keeps
GCS/BigQuery audit artifacts.

Full contracts and recovery behavior are in
`pipelines/GENERATION_PROMOTION.md` and tests in
`pipelines/tests/test_generation_promotion.py`.

## 5. Request consistency

The API queries the singleton active pointer and matching generation-keyed user vector
in one BigQuery statement. It verifies that the active serving binding equals the
container's configured binding, then queries Vertex with the captured generation
restrict. Returned IDs must carry exactly that generation prefix.

If promotion completes while a request is in flight, the request continues on its old,
complete generation. It never combines a new user vector with old items or vice versa.
Missing user vectors, no ranked candidates, or treatment-path failures fall back only
as declared; backend failures return a generic unavailable response.

`tools/export_bundle.py` follows the same principle: read one active generation/epoch,
load exact immutable item/user artifacts from its manifest, and recheck the fence before
writing. Mutable aliases are not used.

## 6. API and experiment data contracts

Cloud Run IAM is private in the deploy workflow, and `/recommend` and `/feedback` also
require `API_BEARER_TOKEN`.

- Body size: at most 2 KiB, JSON object only, unknown fields rejected.
- `/recommend`: bounded `user_id` plus canonical nonzero UUID `request_id`.
- Impression: request UUID is both impression ID and BigQuery streaming insert ID.
  Logging failure prevents exposure/return.
- `/feedback`: bounded user/impression/event plus independent UUID `event_id`.
- Linkage: impression must exist for the same user before feedback is accepted.
- Replay defense: event UUID is the streaming insert ID; downstream marts still
  deduplicate because BigQuery insert IDs are best effort, not a uniqueness constraint.

Assignment is `sha256(user_id) mod 2`, not Python's process-salted `hash()`. Assignment
is therefore stable and at the user—not request—level.

## 7. A/B operating-characteristic validation

`analysis/ab_validation.py` validates the declared experimental analysis with seeded
simulations. The checked synthetic design report includes:

- A/A false-positive rate with an exact Monte Carlo acceptance interval;
- planted-effect confidence-interval coverage;
- power over the predeclared effect grid;
- SRM false-alarm and planted-detection rates;
- production hash-assignment parity;
- a versioned stopping-rule schema/example.

Those simulations validate software and design behavior. The historic synthetic CTR
snapshot remains a wiring reproduction and cannot be presented as a product result or
a preregistered experiment.

## 8. Browser bundle contract

The mock browser uses ranker schema v1. Export and browser scoring share the same
one-hot/binary and label-index/multiclass-softmax contract. Golden fidelity compares
`browser` and `ml_predict` fields; a mock bundle removes/ignores golden data.

The UI labels synthetic state prominently and does not claim sub-millisecond latency,
WASM, production throughput, or commercial lift. The previous remote Ask worker is
disabled; model-controlled trace fields render as text, not HTML.

## 9. Runtime benchmark contract

`benchmarks/vector_search_runtime/runtime_benchmark.py` requires an existing, exact
index/endpoint/deployed-ID/generation binding plus attested non-synthetic vectors and
queries. It validates ADC, a clean commit, at least 100 catalog vectors and 20 queries,
and rejects mixed/unknown generation IDs.

The result schema records:

- raw warmup and measured latencies, requested/success/failure counts, p50/p95;
- exact-vs-ANN top-K overlap, explicitly separate from recommendation metrics;
- resource IDs, object/index generation, promotion epoch, catalog size/dimension;
- client region, OS, CPU, Python, and SDK version.

The harness never creates a Vertex resource and never substitutes local exact search
when the endpoint is unavailable. No result file is checked in because no authorized
meaningful endpoint run has occurred.

## 10. Dependency and build isolation

Python 3.12 environments are separated by purpose. TensorFlow 2.21/TFRS remains on
NumPy 1.26 and patched protobuf 6.33.6. The pipeline pins
`google-cloud-aiplatform==1.165.1` and `google-cloud-storage==3.0.0`, then reasserts the
TensorFlow-compatible NumPy/protobuf pins. Browser/A-B, API/cloud, and MovieLens/Torch
benchmark dependencies are independent.

The historical public build guide is explicitly archived. Current requirement files,
code, and tests are authoritative.

## 11. CI, deployment, and repository security

Pull requests run API/promotion tests, browser schema smoke, A/B tests, MovieLens
benchmark unit/evidence checks, runtime-harness tests, dependency review, and CodeQL.
Actions are pinned to immutable commits and use least permissions.

The GCP deploy workflow is manual, requires `DEPLOY`, uses Workload Identity Federation
and the protected `gcp-production` environment, reads exact Vertex target variables,
and deploys Cloud Run with `--no-allow-unauthenticated`. It does not provision resources
or modify IAM/secrets.

Dependabot, vulnerability alerts, automated security fixes, secret scanning, push
protection, and private vulnerability reporting are enabled. See `SECURITY.md`.

## 12. Remaining limitations

- No GCP resources were changed and no endpoint latency artifact was produced in this
  repository-only upgrade.
- MovieLens is a useful temporal benchmark, not evidence for the synthetic shop domain.
- The benchmark has no frozen cold-item target in its current windows.
- The production training/ranker path still needs a separately validated point-in-time
  real-domain dataset before making a deployed-quality claim.
- Project-level Terraform IAM scopes and cost controls must be reviewed against the
  actual target project before any deployment; repository tests cannot prove live IAM.

## License

MIT; see `LICENSE`.

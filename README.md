# Recommender system: real-data evaluation and generation-safe serving

This repository is a production-shaped recommendation system on Google Cloud:
streaming events, feature tables, two-tower retrieval, Vector Search, reranking,
experiment analysis, and gated retraining. It is a portfolio/reference system—not a
hosted product or a claim of real-world commercial lift.

The repository now separates three kinds of evidence that were previously easy to
confuse:

- **real-data recommendation quality:** frozen MovieLens 1M temporal protocol;
- **synthetic system checks:** simulator, browser mock, and A/B design simulations;
- **runtime evidence:** credentialed benchmark harnesses that emit claims only after an
  actual run against a named resource.

## Current evaluation status

### Real MovieLens validation

The checked benchmark uses the official MovieLens 1M archive pinned by SHA-256. Equal
timestamp blocks are kept together in a global chronological split. Models fit only
past data, mask every previously rated item, rank the same 3,883-item full catalog,
and use exact popularity fallback for cold users.

The validation population has 905 users: 356 warm and 549 cold. Each model family has
one predeclared representative configuration; this is not a hyperparameter sweep or a
state-of-the-art claim.

| Model | NDCG@10 (95% user-bootstrap CI) | Recall@10 | Catalog coverage@10 |
|---|---:|---:|---:|
| popularity | 0.06075 [0.04991, 0.07153] | 0.13702 | 0.02653 |
| item-kNN | 0.05942 [0.04867, 0.06972] | 0.13481 | 0.04636 |
| ALS | **0.06613** [0.05503, 0.07694] | 0.15028 | 0.20577 |
| BPR | 0.06330 [0.05319, 0.07382] | 0.14365 | 0.18491 |
| two-tower ID embeddings | 0.06068 [0.04999, 0.07115] | 0.13702 | 0.13984 |
| ALS retrieval + logistic reranker | 0.06601 [0.05493, 0.07683] | **0.15249** | **0.21401** |

Validation selected ALS. Against the frozen strongest simple baseline (popularity),
the paired validation NDCG@10 difference is `+0.00537`, 95% CI
`[-0.00325, +0.01376]`. That interval crosses zero, so validation did **not** prove a
win.

The one-time test was then run from frozen commit `e67ae1ce`. ALS scored NDCG@10
`0.04540` versus popularity at `0.05398`; the paired difference was `-0.00858`, 95%
CI `[-0.02075, +0.00320]`. The predeclared acceptance gate therefore **failed**. This
is final negative evidence, not a reason to retune against the test. The result and
completed access ledger are committed under `benchmarks/recommender/results/`.

The retrieval+rereanker retrieves exactly 200 unseen candidates before reranking; its
validation target retrieval recall is `0.63204`. Full results, cohorts, NDCG/MRR/Recall
at 10 and 20, 2,000-replicate paired intervals, manifests, and selection receipt are
under [`benchmarks/recommender/`](./benchmarks/recommender/).

### Synthetic evidence is labelled synthetic

The original 300-user/500-item fixture remains useful for wiring checks. Its category
recovery and experiment lift describe a simulator whose preferences were written into
the generator; they are not held-out recommendation quality or business evidence. The
browser demo displays an unmistakable `MOCK / SYNTHETIC` banner and ignores stale
golden files when a mock bundle is active.

The A/B validator checks the analysis design rather than claiming lift. Its seeded
simulations cover A/A false-positive rate, confidence-interval coverage, power across
declared effects, sample-ratio mismatch, deterministic assignment parity, and stopping
rule schema alignment. The checked synthetic design report passes all declared gates;
see [`analysis/ab_validation_report.json`](./analysis/ab_validation_report.json).

## Serving architecture

```text
events -> Pub/Sub -> Dataflow -> BigQuery
                                |-> feature snapshots
                                |-> train immutable user/item generation
                                      |-> evaluation gate
                                      |-> namespaced Vertex datapoints
                                      |-> propagation verification
                                      `-> fenced active-generation pointer

request -> authenticated Cloud Run API
        -> read active generation + matching user vector in one query
        -> Vector Search with generation=<active> restrict
        -> rank candidates -> log impression -> return result
```

Every candidate generation contains canonical item and user artifacts, exact object
generations and SHA-256 hashes, vector dimensions/counts, canaries, model paths, and
the exact Vertex index/index-endpoint/deployed-ID binding.

Promotion is a fenced compensating workflow, not a cross-service transaction. New
namespaced datapoints coexist with old ones. Promotion verifies every candidate
datapoint and its generation restrict through the same deployed endpoint, then moves
the BigQuery pointer/user view/champion together. In-flight requests continue on the
complete generation they captured; user and item embedding spaces are never mixed.

Leases use an owner, monotonic fence epoch, phase, heartbeat, and expiry. Non-stale
concurrent work fails closed; stale recovery and bootstrap use compare-and-set fences.
Cleanup is explicit, dry-run by default, and refuses the active and immediately prior
generations. Full invariants are in
[`pipelines/GENERATION_PROMOTION.md`](./pipelines/GENERATION_PROMOTION.md).

## API security and experiment integrity

`/recommend` and `/feedback` are billable/mutating paths. They require Cloud Run IAM
plus an application bearer token, reject oversized or extra JSON fields, and require
canonical UUID idempotency keys. Recommendations are not returned if impression
logging fails. Feedback must link to an existing impression for the same user and uses
a separate event UUID for replay defense.

The deployment workflow is manual, requires the literal confirmation `DEPLOY`, uses a
protected GitHub environment, and deploys with `--no-allow-unauthenticated`. It does not
create secrets, grant GCP IAM, or provision cost-bearing resources.

## Runtime evidence

No current Vertex endpoint latency is claimed: the repository has not run a meaningful
catalog benchmark against a deployed endpoint in this upgrade. Local fit/scoring times
and browser JavaScript timings are not relabelled as ANN or production latency.

[`benchmarks/vector_search_runtime/`](./benchmarks/vector_search_runtime/) provides a
credentialed, fail-closed harness for an existing endpoint. It requires at least 100
attested non-synthetic catalog vectors and 20 real queries, exact resource/generation
binding, and the production generation filter. A run records every warmup/measured
latency, failures, p50/p95, catalog size/dimension, promotion epoch, client machine and
region, and exact-vs-ANN top-K overlap. Overlap is reported separately from
recommendation quality. The harness never provisions resources or silently falls back
to exact search.

## Reproduce offline checks

Use separate Python 3.12 environments for the cloud application and benchmark:

```bash
# API contracts, promotion fences, mock bundle, and A/B design
python -m pip install -r api/requirements.txt
python -m pytest api/tests -q
python -m unittest -v pipelines.tests.test_generation_promotion
node tools/smoke.mjs
python -m pip install -r analysis/requirements-ab-validation.txt
python -m unittest analysis.tests.test_ab_validation

# Real-data validation evidence (raw MovieLens data remains ignored)
python -m pip install -r benchmarks/recommender/requirements.txt
python -m pytest benchmarks/recommender/tests -q
python benchmarks/recommender/verify_validation_evidence.py
```

The benchmark downloader verifies archive hash
`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.
It fails when the real archive is unavailable and never substitutes synthetic data.

The completed one-time test procedure is documented in
[`benchmarks/recommender/README.md`](./benchmarks/recommender/README.md). It claims a
mode-`0600`, create-once ledger before any test-file read or hash. Concurrent attempts
cannot overwrite the ledger or fixed result path. A failed/crashed attempt remains
consumed and may not be rerun as a first access.

## Repository map

| Path | Purpose |
|---|---|
| `benchmarks/recommender/` | frozen real-data split, six-model comparison, sealing and evidence |
| `benchmarks/vector_search_runtime/` | credentialed endpoint latency/overlap harness; no checked claim yet |
| `training/` | two-tower training and immutable generation manifests |
| `featurestore/` | generation-restricted upsert/visibility and fenced cleanup |
| `pipelines/` | evaluation, lease, promotion, hold, and KFP specification |
| `api/` | authenticated Cloud Run API and exact request/data contracts |
| `analysis/` | synthetic wiring evidence and validated A/B design |
| `demo/`, `tools/` | clearly labelled mock browser bundle, scorer/export schema, smoke checks |
| `infra/`, `dataflow/`, `reco_dbt/` | GCP infrastructure, ingestion, and experiment marts |

[`PUBLIC_BUILD_GUIDE.md`](./PUBLIC_BUILD_GUIDE.md) is an explicitly archived design
narrative and must not be used as the current deployment recipe.

## Known limitations

- MovieLens 1M is real interaction data but small, old, and not representative of an
  online shop. Validation has no cold-item targets under the frozen rule.
- The six model configurations are predeclared representatives, not tuned families.
- The selected ALS model did not beat popularity on the frozen test; it must not be
  promoted or presented as a demonstrated quality improvement.
- No cost-bearing GCP deployment or endpoint benchmark was authorized or performed;
  cloud correctness is covered by contracts/mocks until an operator runs the guarded
  workflow against named resources.
- Production feature joins and the historical BQML ranker recipe still require a
  separately designed point-in-time real-data training path before a production claim.
- The synthetic A/B result validates code and statistical operating characteristics,
  not human behavior, commercial lift, or a deploy decision.

## License

MIT; see [`LICENSE`](./LICENSE).

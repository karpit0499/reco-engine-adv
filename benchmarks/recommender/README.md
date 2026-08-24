# Real-data recommender benchmark

This directory is separate from `analysis/`, whose checked fixture is synthetic and
only validates the cloud pipeline. This benchmark uses MovieLens 1M to evaluate
recommendations at globally ordered future timestamp blocks.

## Frozen protocol

`protocol.json` is authoritative. The source archive is pinned by SHA-256 and raw
MovieLens data is never committed.

- Training uses every event with `timestamp < 975220939` for validation and every
  event with `timestamp < 977083270` for the sealed test.
- Validation targets lie in `[975220939, 977083270)`; test targets lie at or after
  `977083270`. A cutoff assigns the complete equal-timestamp block to the later split.
- A user's target is the lowest movie ID among relevant ratings in that user's first
  timestamp block in the stage window. No later-window event enters fitting data.
- Ratings of 4 or 5 are positive training signals. Candidate masking separately
  excludes every prior rating, including ratings below 4.
- The declared population is the union of warm users with at least five prior
  positives and cold users with no prior observed rating. Users with sparse prior
  history are excluded by the frozen rule.
- Every model ranks the same 3,883-item catalog. Cold items remain candidates. Cold
  users receive a popularity fallback; no random, untrained user embedding is scored.

The checked split manifests record 905 validation users (356 warm, 549 cold) and 961
test users (824 warm, 137 cold). The changing mix is reported rather than hidden.

## Physical split boundary

Only `prepare.py` opens the raw archive. It writes two content-addressed, ignored
artifacts:

- `data/prepared/validation.npz`, which contains validation fitting data and targets
  but no test target rows;
- `data/prepared/test.npz`, which is opened only by the authorized test entry point.

Their committed manifests live in `splits/`. `validate.py` accepts only the validation
artifact and manifest; it has no raw-archive or test-artifact parameter.

This is an auditable process boundary, not cryptographic access control. A repository
owner can inspect or delete local files. The test command mitigates accidental reuse
with exact committed hashes, a fixed result path, and a durably flushed access ledger
created with an atomic `O_CREAT|O_EXCL` claim before the test artifact is read or
hashed. The claim is mode `0600` and carries an unguessable nonce plus a unique process
owner identifier that every later ledger update must preserve.

## Setup and preparation

Use Python 3.12 in a dedicated environment:

```bash
python3.12 -m venv .venv-recommender-benchmark
source .venv-recommender-benchmark/bin/activate
python -m pip install -r benchmarks/recommender/requirements.txt
```

Download, verify, and prepare both deterministic split artifacts:

```bash
python benchmarks/recommender/prepare.py
```

The archive SHA-256 must be
`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.
Use `--replace-prepared` only before the protocol is frozen; identical inputs reproduce
the checked artifact hashes.

## Validation and selection

```bash
python -m pytest benchmarks/recommender/tests -q
python benchmarks/recommender/validate.py --replace-results
```

Validation compares popularity, cosine item-kNN, implicit ALS, BPR, a two-tower ID
embedding model, and an actual two-stage system. The last retrieves 200 unseen items
with ALS, reranks only that pool, and records target retrieval recall.

Each family contributes one predeclared representative configuration. No
hyperparameter sweep was run. “Selected model” therefore means the best of these
declared candidates, not an optimized model family or a state-of-the-art claim.

Recall, NDCG, MRR, and user coverage at 10 and 20 receive 2,000-replicate user
bootstrap intervals. Catalog coverage is a non-additive set statistic, so it is
published as a full-cohort descriptive estimate without a misleading percentile
interval. Warm/cold users and warm/cold target items are reported separately.

`results/selection_receipt.json` hashes the exact validation artifact, validation split,
protocol, config, complete Python source tree, and pinned requirements. Its selected
model and strongest simple baseline are recomputed from the committed validation
artifact during test authorization.

## Freeze, then one test access

Review and commit, in one frozen commit:

- all benchmark Python and tests plus `requirements.txt`;
- `protocol.json`, `configs/validation.json`, and both split manifests;
- `results/validation.json` and `results/selection_receipt.json`.

Keep the ignored prepared artifacts locally; their hashes must match the manifests.
From that exact clean `HEAD`, and only after review, run:

```bash
python benchmarks/recommender/evaluate_test.py \
  --protocol-commit "$(git rev-parse HEAD)"
```

There is no configurable test output. The command refuses an existing or historically
committed `results/test.json` or `results/test_access_ledger.json`, atomically claims
the ledger before reading or hashing the test artifact, and evaluates the selected
model and frozen strongest simple baseline together. Concurrent evaluators cannot
overwrite the claim; exactly one can create it, and the fixed result is also created
without overwrite. A mismatch, exception, or crash consumes the attempt; recoverable
failures update the ledger to `failed` only when its owner and nonce still match. The
acceptance gate passes only when the paired test NDCG@10
confidence interval lower bound is greater than zero. A failed gate remains evidence;
it must not trigger retuning against the test.

The evidence verifier supports both valid repository states: validation-only before
access, and a completed test result plus ledger afterward. In the post-test state CI
checks the result hash, ordinal, frozen commit ancestry, protocol/selection links,
selected-versus-baseline comparison, and acceptance-gate calculation.

Recorded fit and dense full-catalog scoring times are local diagnostics only. They are
not endpoint, browser, ANN, throughput, or production-latency claims.

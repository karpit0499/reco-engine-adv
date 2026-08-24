# Evidence artifacts

`validation.json` contains validation-only model metrics, user-bootstrap intervals,
descriptive catalog coverage, warm/cold cohorts, retrieval diagnostics, environment
versions, and source/split hashes. `selection_receipt.json` hash-links that exact file
and freezes both the selected model and strongest simple baseline.

Neither file proves test performance. `test.json` and `test_access_ledger.json` must not
exist before the frozen protocol commit. The authorized test entry point durably
creates a mode-`0600` ledger with `O_CREAT|O_EXCL` before reading or hashing the
separately prepared test artifact, then evaluates the selected model and baseline in
the same pass. A unique owner identifier and nonce authorize later ledger updates.
Concurrent attempts cannot replace the claim or the fixed result. Hash mismatches and
recoverable failures update the consumed attempt to `failed` rather than reopening it.

The ledger is process-enforced evidence, not cryptographic access control. Repository
owners can alter local state, so preserved Git chronology and checked hashes remain
part of the evidence chain.

CI accepts the valid validation-only state and, after intended access, verifies a
completed result-plus-ledger pair including hashes, ordinal, frozen commit ancestry,
protocol and selection links, and the acceptance-gate calculation.

Catalog coverage is a full-cohort descriptive statistic. Wall-clock fitting and dense
scoring durations are local diagnostics, not endpoint, browser, ANN, throughput, or
production-serving latency.

# Evidence artifacts

`validation.json` contains validation-only model metrics, user-bootstrap intervals,
descriptive catalog coverage, warm/cold cohorts, retrieval diagnostics, environment
versions, and source/split hashes. `selection_receipt.json` hash-links that exact file
and freezes both the selected model and strongest simple baseline.

Those validation files alone do not prove test performance. After the frozen protocol
commit, the authorized ordinal-1 run created `test.json` and
`test_access_ledger.json`. ALS did not beat the frozen popularity baseline and the
declared gate failed. The entry point durably created the mode-`0600` ledger with
`O_CREAT|O_EXCL` before reading or hashing the separately prepared test artifact, then
evaluated both models in the same pass. Its unique owner and nonce authorized later
ledger updates. Concurrent attempts cannot replace the claim or fixed result; a
mismatch or recoverable failure would have consumed the attempt rather than reopening
it.

The ledger is process-enforced evidence, not cryptographic access control. Repository
owners can alter local state, so preserved Git chronology and checked hashes remain
part of the evidence chain.

CI accepts the valid validation-only state and, after intended access, verifies a
completed result-plus-ledger pair including hashes, ordinal, frozen commit ancestry,
protocol and selection links, and the acceptance-gate calculation.

Catalog coverage is a full-cohort descriptive statistic. Wall-clock fitting and dense
scoring durations are local diagnostics, not endpoint, browser, ANN, throughput, or
production-serving latency.

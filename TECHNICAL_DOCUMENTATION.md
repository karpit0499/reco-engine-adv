# Technical Documentation — Real-Time Recommender Engine on GCP

**Audience:** engineers and data scientists evaluating the design. This document assumes fluency with GCP, streaming systems, retrieval models, and experimental statistics, and does not define terms. For a step-by-step build with every term defined, see [`PUBLIC_BUILD_GUIDE.md`](./PUBLIC_BUILD_GUIDE.md).

**Scope:** system design, data contracts, IAM model, model architectures, serving path, experiment design and analysis, drift and retraining, and — at length — the failure modes that shaped the design. The failure-mode sections are the ones worth reading; the rest is table stakes.

---

## 1. System overview

A full-lifecycle recommender: **stream → warehouse → feature store → retrieval model → ANN index → ranker → online API → randomized experiment → statistical analysis → drift monitor → gated retraining → back to the index.**

```
generator ─pub/sub─► Dataflow (Beam, streaming) ─insertAll─► BigQuery.reco.events
                                                                    │
                          hourly scheduled SQL ────────────────────►├─► user_features   (append-only snapshots)
                                                                    └─► item_features   (append-only snapshots)
                                                                            │  registered in
                                                                            ▼
                                                             Vertex AI Feature Registry
                                                          (offline = BigQuery; online = Bigtable, demoed then deleted)
                                                                            │
                      TFRS two-tower ◄──── training pairs (positives from events, latest-value features)
                            │      │
          item vectors ─────┘      └───── user vectors
                │                                │
     GCS (regional mirror)                BigQuery.user_embeddings (append-only, trained_at)
                │                                │
     Vertex AI Vector Search                     │
     (tree-AH, dot product,                      │
      STREAM_UPDATE)                             │
                └──────────────┬─────────────────┘
                               ▼
                    Cloud Run recommender API
              sha256(user_id) % 2 → control | treatment
              treatment: ANN top-50 → BQML ranker → top-10
                               │
              impressions + feedback (append-only, partitioned)
                               │
        ┌──────────────────────┼───────────────────────────┐
        ▼                      ▼                           ▼
  two-proportion z-test   drift queries            KFP v2 retraining pipeline
  (Phase 13)              (PSI-style, Phase 14)    with off-index recall gate (Phase 15.3)
```

**Region and locality.** Compute in `europe-west3`; BigQuery in the `EU` multi-region. This split is the single most load-bearing infrastructure constraint in the build — see §3.2.

---

## 2. Data model

### 2.1 Event stream

Pub/Sub topic `user-events`, JSON payloads. A Beam streaming pipeline (`apache-beam[gcp]==2.75.0`, Streaming Engine enabled) parses, validates, and writes via `WriteToBigQuery` with `insertAll` (streaming inserts, not load jobs).

`reco.events`: `event_id`, `user_id`, `item_id`, `event_type ∈ {view, click, add_to_cart, purchase}`, `ts`.

Event mix is weighted so ≈57 % of events are positive interactions (`click`/`add_to_cart`/`purchase`) — enough signal for the retrieval model without making `view` meaningless.

### 2.2 Feature tables — append-only by design

```
user_features(user_id, feature_timestamp, events_7d, purchases_30d, distinct_items_7d, top_category)
item_features(item_id, feature_timestamp, category, price, popularity_7d)
```

Each scheduled run **INSERTs a new snapshot**, stamped with `feature_timestamp`. Nothing is ever updated in place. Two consequences, both deliberate:

1. **Point-in-time correctness is possible.** `(entity_id, feature_timestamp)` is exactly the shape Vertex's Feature Registry expects, and it is what an as-of join would need. (The training set in this build does **not** yet perform that join — see §9.)
2. **BigQuery's streaming buffer never blocks a write.** Rows in the streaming buffer reject DML for ~90 minutes. Any design that `UPDATE`s freshly streamed rows is a design that intermittently fails in production and works fine in your test. Append-only sidesteps the class entirely.

**The fan-out trap.** Every consumer reads a `*_latest` view (`QUALIFY ROW_NUMBER() OVER (PARTITION BY entity ORDER BY feature_timestamp DESC) = 1`), never the raw table. Joining the raw append-only table duplicates each row once per snapshot — a silent, ratio-preserving fan-out that inflates row counts, breaks nothing visibly, and quietly makes the ranker overconfident. The `_latest` views exist to make the correct path the easy path.

### 2.3 Embeddings

- **Item vectors** → GCS as JSONL (`{"id": ..., "embedding": [32 floats]}`), the Vector Search ingestion format. Overwritten each training run.
- **User vectors** → `reco.user_embeddings(user_id, embedding REPEATED FLOAT, trained_at)`, **append-only**. The API reads `user_embeddings_latest`.

The asymmetry is not an oversight: GCS gives no run history for free, BigQuery does. It is also the reason the demo export (outside this doc's scope) can diff two training runs.

### 2.4 Experiment tables

```
impressions(impression_id, user_id, variant, served_by, served_at)   PARTITION BY DATE(served_at)
feedback(impression_id, event, ts)                                    -- click | conversion
```

Impressions and feedback are **separate append-only tables joined at analysis time**, not one table mutated on click, for the streaming-buffer reason above. `served_by ∈ {personalized, popular}` tags cold-start fallbacks *inside* the treatment arm, which is what makes the intent-to-treat analysis honest (§7.3).

---

## 3. Infrastructure

### 3.1 Terraform ownership boundary

Terraform owns: Pub/Sub topic **and subscription** (declared explicitly, so Dataflow doesn't create its own and force a `pubsub.editor` grant), both GCS buckets, the BigQuery dataset and eight tables, three service accounts and their role bindings.

Terraform deliberately does **not** own: `rank_training`, the BQML `ranker` model, the `*_latest` views, the drift/health views, the Vector Search index and endpoint, or any dbt model. Those are created by SQL or by the `gcloud ai` API. **Every object has exactly one owner.** Declaring a table in both Terraform and SQL means the two fight over its schema on every apply.

`delete_contents_on_destroy = true` on the dataset and `force_destroy = true` on both buckets are what let `terraform destroy` clean up the SQL-created objects it doesn't know about.

### 3.2 Two buckets, and why one won't do

| Bucket | Location | Contents | Consumer |
|--------|----------|----------|----------|
| `…-artifacts` | **`EU`** (multi-region) | Dataflow temp/staging, exported SavedModels, `embeddings/items.json`, archives | everything except Vector Search |
| `…-vectors` | **`europe-west3`** (single region) | a mirror of `embeddings/items.json`, and nothing else, ever | **only** Vector Search, and only at index *creation* |

Vector Search rejects a multi-region `contents_delta_uri`:

```
FAILED_PRECONDITION: The Cloud Storage bucket ... is in location `eu`.
It must be in the same regional location as the service location `europe-west3`.
```

Multi-region `EU` is not "close enough" to regional `europe-west3`. But the artifacts bucket **cannot** simply be pinned to `europe-west3`, because it is Dataflow's staging location, and the escape ladder for a `ZONE_RESOURCE_POOL_EXHAUSTED` stockout (relaunch the job in `europe-west1`/`west4`) only works while that bucket spans all of Europe. One bucket cannot satisfy both constraints. Hence two, with the training script mirroring `items.json` into the regional one on **every** run — because a hand-copied mirror drifts, and a drifted mirror is a silent bug: rebuild the index from a stale mirror and you serve last month's embeddings with no error anywhere.

The `embeddings/` prefix in the mirror must contain **exactly one object**. `contents_delta_uri` reads every object under the prefix; a second file breaks index creation.

### 3.3 IAM — three service accounts, and the three roles nobody expects

| SA | Roles |
|----|-------|
| `df-worker` | `dataflow.worker`, `pubsub.subscriber`, **`pubsub.viewer`**, `bigquery.dataEditor`, `storage.objectAdmin` |
| `trainer` | `aiplatform.user`, `bigquery.dataEditor`, `bigquery.jobUser`, **`bigquery.readSessionUser`**, `storage.objectAdmin`, `artifactregistry.reader`, **`logging.logWriter`** |
| `reco-api` | `aiplatform.user`, `bigquery.dataEditor`, `bigquery.jobUser` |

The bolded three are the ones that cost a debugging session each, and they share a shape: **the code passes as *you* and fails as the *robot*.** Locally you run as Owner; the least-privilege SA is the first identity to actually exercise the permission boundary.

- **`bigquery.readSessionUser`.** `pandas`' `.to_dataframe()` silently routes through the **BigQuery Storage Read API**, a third permission that neither `dataEditor` (table data) nor `jobUser` (run queries) covers. The Vertex training job dies with `403 bigquery.readsessions.create` on the line that reads the training set, while the identical script passes on a laptop.
- **`logging.logWriter`.** Without it, a Vertex AI Pipeline task's **container stdout is silently discarded**. No traceback, no `print()`, nothing — just `The replica workerpool0-0 exited with a non-zero status of 1`, and every log query returns Vertex's own service chatter. You end up debugging the log query instead of the bug. It grants no data access. Add it and forget it.
- **`pubsub.viewer`.** `subscriber` pulls *messages*; it does not read the subscription's *config*, which Dataflow checks at launch. Missing it emits a `PERMISSION_DENIED / GETTING_PUBSUB_SUBSCRIPTION_FAILED` warning at the top of every job log, harmlessly impersonating the real error.

The generalization is the point: **GCP's default-deny model means observability is itself a privilege you must grant.**

### 3.4 CI/CD

GitHub Actions with **Workload Identity Federation**. No service-account JSON key exists anywhere. The OIDC provider carries an attribute condition bound to the exact repository path (`assertion.repository=='<owner>/<repo>'`), and the deployer SA grants `roles/iam.workloadIdentityUser` to the corresponding `principalSet://`. Renaming the repo breaks CI until the attribute condition is updated — which is the trust boundary working, not a bug.

---

## 4. Retrieval — the two-tower model

**Stack:** TensorFlow 2.17 + `tf-keras` + `TF_USE_LEGACY_KERAS=1` + TFRS 0.7.3, Python 3.12.

**Architecture.** Two encoders, `EMB = 32`:

```
user tower:  StringLookup(user_vocab)  → Embedding(|V_u| + 1, 32)
item tower:  StringLookup(item_vocab)  → Embedding(|V_i| + 1, 32)
loss:        tfrs.tasks.Retrieval(metrics=FactorizedTopK(items_ds.batch(256).map(item_model)))
optimizer:   Adagrad(0.1), 5 epochs, batch 512
```

Training pairs are `(user_id, item_id)` for every positive interaction (`click | add_to_cart | purchase`). The towers are ID-embedding-only by design — the point of this build is the *lifecycle*, not a feature-rich tower; adding side features to the towers is a localized change and is left as an obvious extension.

**Cold-start property, stated rather than hidden.** The item vocabulary is built from **interacted** items, not from the catalog. An item nobody has ever clicked gets no embedding, never enters the index, and can never be retrieved. `len(item_ids)` can therefore be `< |catalog|`. This is a real property of two-tower retrieval, and every downstream consumer intersects rather than indexes blindly — otherwise the first short event log produces a `KeyError` in the export path.

**Dimensionality is a contract.** `EMB = 32` must equal the Vector Search index's `dimensions`. A mismatch is rejected at ingestion, not at query time.

**Exported artifacts:** both towers as SavedModels with an explicit string-input signature (nothing in the serving path uses them — the API serves precomputed vectors — but an untested export is a lie in the README), item vectors as JSONL to both buckets, user vectors appended to BigQuery.

### 4.1 The dependency minefield (this is not incidental)

Reproducibility here is load-bearing, and three pins are non-negotiable:

| Problem | Cause | Fix |
|---|---|---|
| A second, older TensorFlow appears | TFRS declares `tensorflow-macos>=2.9 ; sys_platform=="darwin"`, a **dead package** (last release 2.16.2) | install TFRS with `--no-deps` |
| `tf.saved_model.save("gs://…")` fails | TF 2.17 only installs `tensorflow-io-gcs-filesystem` on Python < 3.12 | pin it explicitly |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.5.1`, then `AttributeError: module 'numpy' has no attribute 'long'` from SciPy | `pip install google-cloud-aiplatform` pulls **NumPy 2.x + a NumPy-2-only SciPy + protobuf 6**, and TF 2.17's wheels are compiled against NumPy 1.x | pin `numpy`, `scipy`, `protobuf==4.25.5`; **re-assert the pins immediately after any aiplatform install**, and fail the Docker build if they didn't survive |
| `Could not find a version that satisfies the requirement tensorflow==2.17.0 (from versions: none)` | Python 3.13+. TF 2.17 ships **cp39–cp312 wheels only** | match the interpreter to the pin, not the pin to the interpreter |

Three venvs exist for exactly this reason — `.venv` (TF 2.17: NumPy 1.x, protobuf 4), `.venv-beam` (Beam), `.venv-kfp` (KFP + current aiplatform: NumPy 2, protobuf 6). They pin mutually exclusive versions of the same transitive dependencies. Installing them together silently rewrites TensorFlow's dependency tree out from under it, and the failure surfaces days later in a script nobody touched.

---

## 5. ANN serving — Vertex AI Vector Search

- **Algorithm:** tree-AH (ScaNN). `distance_measure_type = DOT_PRODUCT_DISTANCE`, matching the two-tower objective.
- **`index_update_method = STREAM_UPDATE`** — required, not optional. It is what lets the retraining pipeline `upsert_datapoints()` into the **live** index without a rebuild. Choosing `BATCH_UPDATE` here makes the refresher fail later, at a point where the cause is far away.
- **`public_endpoint_enabled = True`** — without it the endpoint is reachable only over VPC peering, and `find_neighbors()` from Cloud Run or a laptop simply never connects: no useful error, just a timeout.
- **SDK note:** the product was renamed Matching Engine → Vector Search, but the classes are still `MatchingEngineIndex` / `MatchingEngineIndexEndpoint`. On `google-cloud-aiplatform ≥ 1.125`, `create_tree_ah_index()` **only** builds the tree-AH `algorithmConfig` if you explicitly pass `leaf_node_embedding_count` **or** `leaf_nodes_to_search_percent`; pass neither and the SDK sends `algorithmConfig: null`, the null is stripped during Struct serialization, and the API rejects the request with `FAILED_PRECONDITION: algorithmConfig is required but missing from the metadata`. Pass both explicitly.

**Cost:** a deployed index endpoint bills for its node while up. Undeploy between sessions; redeploying takes 20–30 minutes, so do it before you sit down to demo, not during.

---

## 6. Ranking — BQML boosted trees

`CREATE MODEL … OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', input_label_cols=['label'])`.

**Training set** (`rank_training`): positives are observed `(user_id, item_id)` interactions; negatives are uniformly sampled non-interacted pairs from the catalog cross-join, `NOT EXISTS` against the positives.

**Features:** `events_7d`, `purchases_30d`, `distinct_items_7d` (user side); `price`, `popularity_7d` (item side); and the cross feature `same_category = CAST(user.top_category = item.category AS INT64)`.

Two non-negotiables in that query, both silent when violated:

1. **The joins hit `user_features_latest` / `item_features_latest`, never the raw tables** (§2.2 fan-out).
2. **`user_id` and `item_id` are absent from the SELECT list.** BQML treats every non-label column as a feature. Leave the IDs in and the model memorizes them — the classic recsys identity leak, which produces a beautiful AUC and a useless model.

**Validation signal:** `ML.FEATURE_IMPORTANCE` should put `same_category` on top. That is the boosted tree independently rediscovering the same hidden preference structure the two-tower model found, through an entirely different lens. Agreement between two independent models on the same latent structure is a stronger correctness signal than either model's own metric.

**Why two stages at all:** stage 1 (cheap, approximate, high recall) discards ~90 % of the catalog in milliseconds; stage 2 (expensive, precise, high precision) carefully orders what survives. Applying stage-2 features to the whole catalog does not scale; applying stage-1 geometry alone does not rank. The split is the architecture.

---

## 7. The online experiment

### 7.1 Assignment

`variant = "treatment" if int(sha256(user_id).hexdigest(), 16) % 2 else "control"` — computed in the API, stateless, deterministic, and **per-user**, not per-request. A user's arm never flips mid-session, and no assignment table needs to exist.

### 7.2 Serving path

- **control:** top-N by `popularity_7d`. A real baseline, not a random one — beating "popular items" is the bar that matters, and it is a bar many personalization systems fail.
- **treatment:** ANN `find_neighbors(user_vector, k=50)` → hydrate features → `ML.PREDICT` on the ranker → top-10 by predicted probability.
- **fallback:** a user with no embedding (new user, or trained before their first event) is served the popular list and tagged `served_by = 'popular'` **while remaining in the treatment arm**.

### 7.3 Analysis

Two-proportion z-test on CTR, with a 95 % confidence interval on the absolute difference. Reported as a full statement — lift %, absolute difference, CI, z, p — never as a bare percentage.

**Intent-to-treat.** Fallback-served impressions stay in the treatment arm. Dropping them would measure "personalization, given that personalization was possible," which is not the decision anyone actually faces. Instead, the fallback rate is reported alongside the lift: it is *part* of the treatment's real-world performance, and it is the number that tells you what fraction of the lift is being left on the table by cold start.

**Caveats that are stated before anyone asks:**
- **Peeking.** The test is run once, at a pre-decided stopping point. Refreshing a z-test until it crosses 0.05 manufactures false positives at a rate far above the nominal α.
- **The click model is an assumption.** Simulated users click according to a rule I wrote. The *method* generalizes; the *number* does not.
- **Independence.** The z-test assumes independent impressions. Repeated impressions to the same user are correlated; at this scale it's a footnote, at production scale it's a cluster-robust variance estimate.
- **Different CI formulae disagree.** The Newcombe interval (`statsmodels`) and Agresti–Caffo differ in the third decimal on small samples. If two implementations must agree, say which one is normative.

---

## 8. Drift & retraining

### 8.1 Monitoring

- **Model health:** CTR per variant per day. A live-lift regression shows up here first.
- **Feature drift:** distributional comparison of `user_features` snapshots against a baseline window (mean/variance shift on the numerics, category-share shift on `top_category`). With a single snapshot the query returns **rows of zeros, not zero rows** — the difference matters if you alert on emptiness.

Vertex AI Model Monitoring is the managed equivalent, but it monitors **endpoint-served** models. This system's model surface is an **ANN index plus a BQML model**, neither of which is a Vertex endpoint — so the managed product does not apply, and the SQL checks are the honest implementation of the same instinct against the architecture that actually exists.

### 8.2 The retraining pipeline (KFP v2 on Vertex AI Pipelines)

```
train → evaluate (off-index, brute force) → dsl.If(recall_ok) ─► refresh_index (upsert_datapoints)
                                          └ dsl.Else ─────────► hold (fail loudly)
```

**The gate must evaluate off-index. This is the subtle part.**

The obvious design — "retrain, then score the new model against the live index; promote if recall improved" — **can never open the gate.** Immediately after a retrain you have *new user vectors* and, in the live index, *old item vectors*. Scoring one against the other measures the **consistency window**, not the model, and returns ≈chance after every retrain, good or bad. A gate wired to that signal is a gate that is permanently shut, and it will look like the model is permanently broken.

So the evaluator brute-forces the **fresh** item vectors against the **fresh** user vectors (dot product, matching the index's distance measure), computes top-K recall against ground truth, and compares against a **champion score persisted in GCS**.

**The gate rule is "don't promote a regression," not "only promote an improvement."** Strict improvement sounds rigorous and is a trap: once the model saturates on the available data, a perfectly healthy retrain lands *within noise* of the previous run, and a strict rule means you never promote again. The rule is: pass if the score clears an absolute floor **and** has not dropped more than a small margin below the reigning champion.

**What the gate does not protect** (say this before someone finds it): `train.py` writes user vectors to BigQuery **unconditionally**. A *closed* gate therefore still leaves the system mid-window — new user vectors, old item vectors — until the next successful promotion. Closing that hole properly means staging user vectors behind the same gate and swapping index + user table atomically via **index aliasing**. It is a known gap, not a discovered one.

**KFP operational notes:**
- Every task sets `set_caching_options(False)`. A component with no inputs has a constant cache key, and Vertex will cheerfully "succeed" by skipping training entirely.
- The pipeline image installs `google-cloud-aiplatform` (for `upsert_datapoints`) *after* TensorFlow, then **re-asserts the NumPy/SciPy/protobuf pins** and **fails the build** if they didn't survive (§4.1). Better a red build than a 3 a.m. pipeline failure with a NumPy ABI error in the traceback.

---

## 9. Known limitations

- **Training-set features are latest-value, not point-in-time.** The append-only snapshots make an as-of join possible; the join isn't done. This leaks a small amount of future information into the ranker's training rows.
- **The recall gate covers item vectors only** (§8.2).
- **Cold-start items are unretrievable** (§4) — no content-based tower, so a new item requires interactions before it can be recommended.
- **The API is unauthenticated**, so the load simulator can reach it without auth plumbing. Production: IAP or an API gateway.
- **Impressions are treated as independent** in the z-test (§7.3).
- **The population is simulated.** The evaluation validates the pipeline, not a business hypothesis.
- **Online feature serving is not on the serving path.** Bigtable is created, benchmarked, and deleted, because it has a hard **per-node-hour** cost floor (published figures have ranged ~\$0.65–1.20/node-hour ⇒ roughly €450–850/month for one node, left up). Development serves features from BigQuery directly. The trade-off is the deliverable; the Bigtable instance is not.

## 10. Extensions, in priority order

1. **Index aliasing + staged user vectors** — closes the consistency window and makes the gate total rather than partial.
2. **Point-in-time feature retrieval** for training-set construction.
3. **Real-time user tower** on a Vertex endpoint: session-fresh embeddings instead of retrain-fresh ones.
4. **Sequence models** (transformer-based session recommendation) as a retrieval alternative.
5. **Multi-armed bandits** replacing the fixed 50/50 split — stop paying for the losing arm once the posterior is decisive.
6. **Scheduled retraining** on Cloud Scheduler with the gate in the loop.
7. **Bigtable online serving at real QPS**, with the cost justified by the traffic rather than by the demo.

---

## Appendix A — Reproducing the numbers

| Claim | Produced by |
|---|---|
| CTR lift, CI, z, p | `analysis/ab_test.py` (Phase 13) |
| Preferred-category share in top-10 vs 20 % random baseline | `featurestore/test_retrieval.py` (Phase 9) |
| `same_category` tops feature importance | `ML.FEATURE_IMPORTANCE(MODEL reco.ranker)` (Phase 10) |
| Bigtable cold vs warm read latency (report as a **pair**, with `n`) | `featurestore/online_store_demo.py` (Phase 6.2) |
| Recall gate open/closed decision | the `evaluate` component's KFP output (Phase 15.3) |

## Appendix B — Cost surface

| Resource | Bills when | Off-switch |
|---|---|---|
| Dataflow streaming job | always, while running | `gcloud dataflow jobs drain <ID> --region …` |
| Vector Search deployed index | always, while deployed | `gcloud ai index-endpoints undeploy-index …` |
| Bigtable online store | **always, at a per-node-hour floor** | deleted in a `finally` block; verify with the Feature Store SDK — `gcloud bigtable instances list` shows an **empty list while it is billing**, because the instance lives in Google's project, not yours |
| Cloud Run, Pub/Sub, BigQuery storage, BQML, one-shot training | scale-to-zero or pennies | — |

`terraform destroy` removes the substrate. It does **not** remove the Vector Search index or endpoint — those live outside Terraform and must be deleted first.

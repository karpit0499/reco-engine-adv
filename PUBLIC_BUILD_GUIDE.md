# Real-Time Recommender Engine on GCP — Public Build Guide

**A complete, reproducible, beginner-friendly guide to building a production-shaped recommender system on Google Cloud, from an empty project to a running A/B test with a gated retraining pipeline.**

Every command, every file, and every gotcha is written out in full. Nothing is left as an exercise. The guide was executed end-to-end on a clean GCP project, and the failures encountered along the way are documented *inline, at the step where they fire* — because a build guide that only shows the happy path teaches you nothing about the day the happy path stops working.

---

## What this builds

User interactions stream through **Pub/Sub** and **Dataflow** into BigQuery. Append-only feature snapshots are cataloged in the (current, non-legacy) **Vertex AI Feature Store**. A **two-tower retrieval model** (TensorFlow Recommenders) learns user and item embeddings. **Vertex AI Vector Search** serves nearest-neighbour candidates in milliseconds; a **BQML boosted-tree ranker** re-orders them. A live **A/B test with a real significance test** measures lift over a popular-items baseline. Drift monitoring and a **gated retraining pipeline** — which refuses to promote new embeddings into the live index unless offline recall holds up — keep it healthy. Everything is Terraform-provisioned and shipped by keyless GitHub Actions.

Two things here are unusual for a self-directed build: **a feature store framed around training–serving skew**, and **an A/B test analyzed with statistics rather than vibes**.

## Who this is for

- You can use a terminal and you have a Google account. That's the floor.
- You do **not** need prior GCP, Terraform, Beam, or TensorFlow experience — every term is defined the first time it appears.
- **The commands assume macOS** (Homebrew for Python, Apple Silicon for the Docker architecture gotcha in Step 7.4). On Linux everything works with the obvious substitutions — your package manager for `brew`, and the `linux/amd64` build note in Step 7.4 stops applying because you are already on it.
- Budget: this runs at roughly **€0–5** if you follow the off-switches. Two resources bill *while switched on* (a Dataflow streaming job, a deployed Vector Search index endpoint) and one bills *heavily* (Bigtable online serving, hundreds of € / month for a single node). The guide tells you exactly when to kill each one.

> **How to use this guide.** **Phases 0–7 are the Core Path** — event stream, features, and a trained two-tower model producing recommendations. **Phases 8–15 are the production system** — vector serving, ranking, the online API, the experiment, the statistics, drift, and retraining. **Phases 16–18** add dbt, CI/CD, and packaging. One phase per sitting; every phase ends with a ✅ checkpoint. Do not try to do this in one weekend.

> **Safety note before you start.** Nothing in this guide asks you to download a service-account key, and you should not create one. Authentication is `gcloud auth application-default login` locally and **Workload Identity Federation** in CI. Every `.tfstate`, key file, and generated data dump is `.gitignore`d in Phase 17. Substitute your own project ID and GitHub username wherever the guide shows them.

## Architecture

```
   event generator (users with hidden category preferences)
                     │  publish
                     ▼
        ┌──────────────────────┐      ┌───────────────────────────┐
        │  Pub/Sub             │ ───► │  Dataflow (Apache Beam)    │
        │  user-events topic   │      │  streaming ingestion       │
        └──────────────────────┘      └────────────┬──────────────┘
                                                   ▼
        ┌───────────────────────────────────────────────────────────┐
        │  BigQuery                                                  │
        │   events · catalog · user_features · item_features         │
        │   (append-only snapshots = the OFFLINE FEATURE STORE)      │
        └───────┬───────────────────────────────────┬───────────────┘
                │ registered in                     │ training pairs
                ▼                                   ▼
   ┌──────────────────────────┐      ┌───────────────────────────────┐
   │ Vertex AI Feature Store  │      │  Two-tower model (TFRS)        │
   │ Feature Registry         │      │  user tower · item tower       │
   └──────────────────────────┘      └───────┬───────────────┬───────┘
                                             │ item vectors  │ user vectors
                                             ▼               ▼
                              ┌─────────────────────┐  ┌──────────────────┐
                              │ Vertex AI           │  │ BigQuery         │
                              │ Vector Search (ANN) │  │ user_embeddings  │
                              └──────────┬──────────┘  └────────┬─────────┘
                                         │                      │
                                         └──────────┬───────────┘
                                                    ▼
                 ┌────────────────────────────────────────────────┐
                 │  Recommender API (Cloud Run)                    │
                 │  deterministic hash → 50/50 split               │
                 │    control   → popular items                    │
                 │    treatment → retrieve 50 → BQML rank → top-10 │
                 └───────────────────┬────────────────────────────┘
                                     │ impressions + feedback (append-only)
                                     ▼
        z-test lift analysis · drift monitor · retrain → live index upsert
                                     ▼
                       dbt experiment marts + Looker Studio

  Terraform-provisioned · keyless GitHub Actions · nothing left running idle
```

## Skills this build exercises

| Skill | Where it appears | Why it matters |
|-------|------------------|----------------|
| IaC (Terraform) | Phase 1 | Very high |
| Streaming ingestion (Pub/Sub + Dataflow) | Phase 2–3 | Very high |
| Feature engineering in the warehouse | Phase 4 | High |
| **Feature Store + training-serving skew** | Phase 5–6 | **Very high — rarely done properly** |
| Two-tower retrieval (TFRS) | Phase 7 | **High — recsys core** |
| **Vector Search (ANN retrieval)** | Phase 8–9 | **High — recsys/GenAI** |
| Two-stage retrieve→rank architecture | Phase 10 | High |
| Low-latency online serving | Phase 11 | High |
| **A/B testing + significance testing** | Phase 12–13 | **Very high — rarely done properly** |
| Drift monitoring + retraining loop | Phase 14–15 | Very high — MLOps |
| **Pipeline orchestration + conditional deploy gate (KFP / Vertex AI Pipelines)** | Phase 15.3 | **Very high — MLOps** |
| dbt + BI | Phase 16 | High |
| CI/CD (keyless WIF) | Phase 17 | High |

> **Version & product notes — read before building:**
> - **Vertex AI Feature Store** current design: your features live in **BigQuery (the offline store)**; a **Feature Registry** (feature groups → features) catalogs them; **Bigtable online serving** is the supported low-latency path. The **legacy Feature Store** (`aiplatform.Featurestore` / `EntityType` API in older tutorials) is **deprecated and sunsets 17 Feb 2027** — don't build on it. **Optimized online serving is also deprecated (May 2026).**
> - **⚠️ Bigtable online serving has a hard cost floor.** Google's Vertex AI pricing page lists Bigtable online serving at a **per-node-hour** rate (published figures have ranged from about **$0.65 to $1.20 per node-hour**, so at one node that is roughly **€450–850/month** if you leave it up). Check [the current Vertex AI pricing page](https://cloud.google.com/vertex-ai/pricing) before you create one — but the number you need to remember is *"hundreds of euros a month, minimum, for one node."* This guide serves features **from BigQuery directly** for development, and demonstrates Bigtable **briefly, then deletes it in a `finally` block**. Knowing where that cost floor sits, and deciding against it on purpose, is the whole point of the phase.
> - **Vector Search** is the current name for Matching Engine. The **SDK classes are still called `MatchingEngineIndex` / `MatchingEngineIndexEndpoint`** — verified present and working on `google-cloud-aiplatform` **1.160.0**, so the code below is current, not legacy. A **deployed index endpoint bills while up** — undeploy between sessions.
> - **TensorFlow ≥ 2.16 defaults to Keras 3, which breaks TFRS.** The fix used throughout: install `tf-keras` and set `TF_USE_LEGACY_KERAS=1` *before* importing TensorFlow.
> - **⚠️ `pip install tensorflow-recommenders` will try to install `tensorflow-macos` on a Mac.** TFRS declares `tensorflow-macos>=2.9.0 ; sys_platform == "darwin"`, and **`tensorflow-macos` is a dead package — its last release was 2.16.2**. On Apple Silicon, a plain `pip install -r training/requirements.txt` therefore drops a *second, older* TensorFlow on top of the one you asked for, and you spend an evening on import errors that make no sense. Phase 7 installs TFRS with `--no-deps` for exactly this reason. This is the single most likely thing to bite you on this machine.
> - Pin-check before building: `tensorflow-recommenders`, `google-cloud-aiplatform`, and the Feature Store preview import path all move frequently. The **concepts** below are stable even when class names get reshuffled.
> - Google is currently rebranding parts of Vertex AI under **"Gemini Enterprise Agent Platform"**, so Feature Store doc URLs move. The APIs called here are unchanged.

## Conventions used throughout

- Project **`reco-engine-adv`**, region **`europe-west3`**, BigQuery multi-region **`EU`**.
- **Two buckets, and the difference between them matters:**

  | Bucket | Location | Holds | Read by |
  |--------|----------|-------|---------|
  | `reco-engine-adv-artifacts` | **`EU`** (multi-region) | Dataflow temp/staging, saved towers, `embeddings/items.json`, `archive/` | everything except Vector Search |
  | `reco-engine-adv-vectors` | **`europe-west3`** (single region) | a mirror of `embeddings/items.json` — nothing else, ever | **only** Vector Search, and only at index *creation* |

  This looks like redundancy. It isn't. **Vector Search refuses a multi-region bucket** — `contents_delta_uri` must live in a bucket whose location string is *exactly* the index's region. And the artifacts bucket cannot simply be moved to `europe-west3`, because it is Dataflow's staging location and Step 3.3's escape ladder ("if Frankfurt is out of machines, run the job in `europe-west1`") **only works while that bucket is EU-wide**. One bucket cannot satisfy both constraints, so there are two.
- **Region caveat:** Vector Search and Feature Store online serving are best-documented in `us-central1` / `europe-west4`. Target `europe-west3` first; if a specific resource 404s regionally, fall back to `europe-west4` *for that resource only* and keep BigQuery in the EU.
- **Three venvs** — Beam, TensorFlow, and KFP each pin conflicting `protobuf`/`grpcio`/`numpy` versions, so they never share an environment:

  ```bash
  mkdir -p ~/dev/reco-engine-adv && cd ~/dev/reco-engine-adv
  brew install python@3.12                          # skip if you already have 3.12
  /opt/homebrew/bin/python3.12 --version            # MUST print 3.12.x — see the box below
  /opt/homebrew/bin/python3.12 -m venv .venv        # SDKs, TFRS training, analysis, dbt
  /opt/homebrew/bin/python3.12 -m venv .venv-beam   # ONLY Apache Beam
  # .venv-kfp is created later, in Step 15.3 — you don't need it until then
  ```

  Note the **explicit interpreter path**. `python3 -m venv` inherits whatever `python3` currently
  resolves to — which on a current Mac is 3.13 or newer, and that is a wall you hit later, in
  Step 7.1, with an error message that doesn't say "wrong Python."

  | venv | Used by |
  |------|---------|
  | `.venv` | everything: generator, Feature Store, training, API, analysis, dbt |
  | `.venv-beam` | **only** `dataflow/stream_events.py` |
  | `.venv-kfp` | **only** compiling and submitting the Step 15.3 pipeline (created in that step) |

  The third one is not fussiness. `.venv` holds TensorFlow 2.17, which needs **NumPy 1.x** and
  **protobuf 4**; `kfp` and a current `google-cloud-aiplatform` want **NumPy 2** and **protobuf 6**.
  Installing them together silently rewrites TensorFlow's dependencies out from under it — a failure
  that surfaces days later, in a script you didn't touch. Step 7.4 already forces you to walk that
  tightrope once (with a re-pin line); Step 15.3 declines to walk it a second time.

> **⚠️ Python-version gotcha — check this *before* you create the venvs.** **TensorFlow 2.17 caps out at Python 3.12**, and TF 2.17 is the last version that pairs cleanly with `tf-keras` + TFRS. TF 2.17 publishes wheels for **cp39–cp312 only** (the macOS-arm64 ones exist and are native — `tensorflow-2.17.0-cp312-cp312-macosx_12_0_arm64.whl`). On Python 3.13+ there is no wheel at all, and `pip install -r training/requirements.txt` fails in Step 7.1 with:
>
> ```
> ERROR: Could not find a version that satisfies the requirement tensorflow==2.17.0 (from versions: none)
> ERROR: No matching distribution found for tensorflow==2.17.0
> ```
>
> **`from versions: none` does not mean "2.17.0 was withdrawn."** It means pip found *zero* TensorFlow releases installable on this interpreter — i.e. your Python is too new. Bumping the pin is the wrong reflex: `tensorflow-recommenders` is built against the 2.1x line, so raising TF cascades into the model code. **Match the interpreter to the pin, not the other way round.**
>
> Use **Python 3.12** for both venvs (Beam 2.75 is happy on 3.10–3.14, so 3.12 satisfies both and keeps the two environments interchangeable).
>
> **If you already created a venv on 3.13+**, don't try to repair it — rebuild it:
> ```bash
> deactivate                                        # leave the bad venv
> rm -rf .venv .venv-beam
> brew install python@3.12
> /opt/homebrew/bin/python3.12 -m venv .venv
> /opt/homebrew/bin/python3.12 -m venv .venv-beam
> source .venv/bin/activate
> python --version                                  # must now say 3.12.x
> pip install --upgrade pip
> ```
> Then re-run whatever `pip install` you were on. (No Homebrew? The python.org macOS ARM64 installer for 3.12 works identically — after it, plain `python3.12 -m venv .venv` is on your PATH.)

- **Command blocks vs file blocks:** `gcloud` / `git` / `bq` / `curl` blocks are terminal commands; Python / SQL / HCL / YAML blocks are file contents (or the BigQuery query editor).

---

# CORE PATH

# Phase 0 — Foundations (~45 min)

**Goal:** Project, billing, budget, and a clear picture of the three things that cost money.

## Step 0.1 — Check your tools

```bash
gcloud --version
terraform -version
python3 --version        # 3.10–3.12 (see the version gotcha above)
gh --version
```

## Step 0.2 — Create the project

```bash
gcloud auth login
gcloud auth application-default login          # lets local Python SDKs use your credentials

gcloud projects create reco-engine-adv --name="Recommender Engine Advanced"
gcloud config set project reco-engine-adv
gcloud billing accounts list
gcloud billing projects link reco-engine-adv --billing-account=XXXXXX-XXXXXX-XXXXXX
gcloud billing projects describe reco-engine-adv    # confirm billingEnabled: true
```

> **⚠️ `ALREADY_EXISTS: Requested entity already exists` on the very first command is expected.** GCP project IDs are unique across *all of Google Cloud*, not just your account — and `reco-engine-adv` is taken, by the project this guide was written against. **Pick your own ID** (`reco-engine-adv-<something>`) and use it everywhere the guide says `reco-engine-adv`. The same is true of the two GCS bucket names in Phase 1, which are also a global namespace.
>
> Setting it once, here, saves you finding it 28 times later:
>
> ```bash
> export PROJECT_ID=reco-engine-adv-<something>     # your unique ID
> ```
>
> The guide writes the literal string `reco-engine-adv` in code blocks rather than `$PROJECT_ID`, because a literal is easier to read and to debug. If you change the ID, the honest way to keep up is a single find-and-replace across your repo before you start Phase 1 — the string appears in the Terraform, the Beam launch, `train.py`, the API, the dbt profile, and the CI workflow.

## Step 0.3 — Budget alert (your safety net)

Console → **Billing → Budgets & alerts** → **€15** scoped to `reco-engine-adv`, with 50/90/100 % email alerts.

> **The cost map — memorize these three.** Things that bill *while switched on*: (1) the **Dataflow streaming job** (single-digit €/day), (2) the **Vector Search deployed index endpoint** (single-digit €/day), (3) the **Bigtable online store** (per-node-hour, ~€0.50–1.00/hr — the expensive one, and only ever on for a short demo). Everything else is scale-to-zero or free-tier. The final section has the exact off-switches; the budget alert is your backstop.

✅ **Checkpoint 0:** project exists, billing enabled, budget set — and you can name the three resources that cost money.

---

# Phase 1 — Provision with Terraform (~2 hrs)

**Goal:** Declare the entire substrate — topic, subscription, bucket, eight BigQuery tables, three service accounts — in code.

> **Term — Infrastructure as Code (IaC):** defining cloud resources in versioned text files instead of clicking a console. Reproducible, reviewable, and deletable in one command (`terraform destroy`).

```bash
mkdir -p ~/dev/reco-engine-adv/infra && cd ~/dev/reco-engine-adv/infra
```

**`infra/main.tf`** (long, but it's the whole substrate):

```hcl
terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 7.0" }
  }
}

provider "google" {
  project = "reco-engine-adv"
  region  = "europe-west3"
}

resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",            # Vertex AI: Feature Store, Vector Search, training
    "pubsub.googleapis.com",
    "dataflow.googleapis.com",
    "compute.googleapis.com",               # Dataflow workers ARE Compute Engine VMs — without
                                            # this, the Phase 3 launch fails on worker startup
    "bigquery.googleapis.com",
    "bigquerydatatransfer.googleapis.com",  # BigQuery SCHEDULED QUERIES run on this API
    "bigtable.googleapis.com",              # Phase 6.2 online-store demo
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",                   # service accounts + IAM bindings below
    "monitoring.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ── Streaming ───────────────────────────────────────────────────────────
resource "google_pubsub_topic" "events" {
  name       = "user-events"
  depends_on = [google_project_service.apis]
}

# Declare the subscription explicitly. Pointing Dataflow at a *topic* makes it
# create its own subscription, which forces you to grant pubsub.editor —
# declaring it here keeps the worker on pubsub.subscriber (least privilege).
resource "google_pubsub_subscription" "events_df" {
  name                 = "events-df-sub"
  topic                = google_pubsub_topic.events.name
  ack_deadline_seconds = 60
}

# EU = a MULTI-REGION bucket. Deliberate: it is Dataflow's temp/staging location,
# and Step 3.3's escape ladder (relaunch the job in europe-west1/west4 when
# Frankfurt is out of machines) only works because this bucket spans all of
# Europe. Do NOT pin it to europe-west3 to make Vector Search happy — see below.
resource "google_storage_bucket" "artifacts" {
  name                        = "reco-engine-adv-artifacts" # globally unique — suffix if taken
  location                    = "EU"
  uniform_bucket_level_access = true
  force_destroy               = true
  depends_on                  = [google_project_service.apis]
}

# Vector Search reads contents_delta_uri from a bucket whose location is the
# EXACT SAME SINGLE REGION as the index. Multi-region "EU" is rejected outright:
#   FAILED_PRECONDITION: The Cloud Storage bucket ... is in location `eu`.
#   It must be in the same regional location as the service location `europe-west3`.
# Hence this second, tiny, REGIONAL bucket. train.py (Phase 7) mirrors items.json
# into it; Phase 8 points the index here. Nothing else ever reads it.
resource "google_storage_bucket" "vectors" {
  name                        = "reco-engine-adv-vectors" # globally unique — suffix if taken
  location                    = "europe-west3"            # REGIONAL. Not "EU".
  uniform_bucket_level_access = true
  force_destroy               = true
  depends_on                  = [google_project_service.apis]
}

# ── BigQuery ────────────────────────────────────────────────────────────
# delete_contents_on_destroy = true is REQUIRED: later phases create tables,
# views and a BQML model in this dataset via SQL (they are deliberately NOT
# declared here). Without this flag, `terraform destroy` fails on a non-empty
# dataset and you're left hand-deleting things in the console.
resource "google_bigquery_dataset" "reco" {
  dataset_id                 = "reco"
  location                   = "EU"
  delete_contents_on_destroy = true
  depends_on                 = [google_project_service.apis]
}

resource "google_bigquery_table" "events" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "events"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "event_time"
  }

  schema = jsonencode([
    { name = "event_id",   type = "STRING" },
    { name = "user_id",    type = "STRING" },
    { name = "item_id",    type = "STRING" },
    { name = "event_type", type = "STRING" },     # view | click | add_to_cart | purchase
    { name = "event_time", type = "TIMESTAMP" },
  ])
}

# Ground truth from the generator (loaded once with `bq load --replace`, Phase 2)
resource "google_bigquery_table" "catalog" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "catalog"
  deletion_protection = false

  schema = jsonencode([
    { name = "item_id",  type = "STRING" },
    { name = "category", type = "STRING" },
    { name = "price",    type = "FLOAT" },
  ])
}

resource "google_bigquery_table" "users_truth" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "users_truth"
  deletion_protection = false

  schema = jsonencode([
    { name = "user_id",            type = "STRING" },
    { name = "preferred_category", type = "STRING" },
  ])
}

# Feature tables — these ARE the Feature Store's offline store.
# APPEND-ONLY: Phase 4 INSERTs a new snapshot per run, stamped with
# feature_timestamp. That column is what makes point-in-time joins possible
# and is exactly the shape Vertex's Feature Registry expects.
resource "google_bigquery_table" "user_features" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "user_features"
  deletion_protection = false

  schema = jsonencode([
    { name = "user_id",           type = "STRING" },
    { name = "feature_timestamp", type = "TIMESTAMP" },
    { name = "events_7d",         type = "INTEGER" },
    { name = "purchases_30d",     type = "INTEGER" },
    { name = "distinct_items_7d", type = "INTEGER" },
    { name = "top_category",      type = "STRING" },
  ])
}

resource "google_bigquery_table" "item_features" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "item_features"
  deletion_protection = false

  schema = jsonencode([
    { name = "item_id",           type = "STRING" },
    { name = "feature_timestamp", type = "TIMESTAMP" },
    { name = "category",          type = "STRING" },
    { name = "price",             type = "FLOAT" },
    { name = "popularity_7d",     type = "INTEGER" },
  ])
}

# Batch-computed user embeddings (written by training, read by the API)
resource "google_bigquery_table" "user_embeddings" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "user_embeddings"
  deletion_protection = false

  schema = jsonencode([
    { name = "user_id",    type = "STRING" },
    { name = "embedding",  type = "FLOAT", mode = "REPEATED" },
    { name = "trained_at", type = "TIMESTAMP" },
  ])
}

# A/B experiment: impressions are APPEND-ONLY…
resource "google_bigquery_table" "impressions" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "impressions"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "served_at"
  }

  schema = jsonencode([
    { name = "impression_id", type = "STRING" },
    { name = "user_id",       type = "STRING" },
    { name = "variant",       type = "STRING" },   # control | treatment
    { name = "served_by",     type = "STRING" },   # personalized | popular  (cold-start fallback)
    { name = "items",         type = "STRING" },   # JSON array of item_ids
    { name = "served_at",     type = "TIMESTAMP" },
  ])
}

# …and clicks/conversions are SEPARATE feedback events, joined at analysis
# time. (Never UPDATE freshly streamed rows: BigQuery's streaming buffer
# blocks DML on them for ~90 minutes — a lesson worth
# learning once, in a place that doesn't page you.)
resource "google_bigquery_table" "feedback" {
  dataset_id          = google_bigquery_dataset.reco.dataset_id
  table_id            = "feedback"
  deletion_protection = false

  schema = jsonencode([
    { name = "impression_id", type = "STRING" },
    { name = "user_id",       type = "STRING" },
    { name = "event",         type = "STRING" },   # click | conversion
    { name = "at",            type = "TIMESTAMP" },
  ])
}

# ── Service accounts ────────────────────────────────────────────────────
resource "google_service_account" "dataflow" {
  account_id   = "df-worker"
  display_name = "Dataflow Worker"
  depends_on   = [google_project_service.apis]
}

resource "google_service_account" "trainer" {
  account_id   = "trainer"
  display_name = "Training / Pipelines"
  depends_on   = [google_project_service.apis]
}

resource "google_service_account" "api" {
  account_id   = "reco-api"
  display_name = "Recommender API"
  depends_on   = [google_project_service.apis]
}

# dataEditor = read/write table DATA; jobUser = permission to RUN queries.
# The trainer and the API both run queries, so they need both. Dataflow only
# streams rows in (insertAll), never runs a query — so no jobUser for it.
# artifactregistry.reader on the trainer is what lets Vertex AI pull your
# training image in Step 7.4; without it the job fails on an image pull.
# readSessionUser on the trainer is the third, separate BigQuery permission —
# see the box below. Without it, Step 7.4's Vertex job dies on line 26 of
# train.py with a 403, while the identical script passes on your laptop.
# logging.logWriter is what lets the trainer's containers WRITE LOGS AT ALL.
# Omit it and Step 15.3's pipeline fails with a bare "exit status 1" and an
# empty log — see the box below it. It grants no data access; add it and forget it.
locals {
  df_roles = [
    "roles/dataflow.worker",
    "roles/pubsub.subscriber",
    "roles/pubsub.viewer", # see note below — without it the job WARNs on every launch
    "roles/bigquery.dataEditor",
    "roles/storage.objectAdmin",
  ]
  trainer_roles = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/bigquery.readSessionUser", # .to_dataframe() → Storage Read API. Not covered by the two above.
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter", # WITHOUT THIS, CONTAINER STDOUT IS DISCARDED. See the box below.
  ]
  api_roles = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
  ]
}

resource "google_project_iam_member" "df" {
  for_each = toset(local.df_roles)
  project  = "reco-engine-adv"
  role     = each.value
  member   = "serviceAccount:${google_service_account.dataflow.email}"
}

resource "google_project_iam_member" "trainer" {
  for_each = toset(local.trainer_roles)
  project  = "reco-engine-adv"
  role     = each.value
  member   = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_project_iam_member" "api" {
  for_each = toset(local.api_roles)
  project  = "reco-engine-adv"
  role     = each.value
  member   = "serviceAccount:${google_service_account.api.email}"
}
```

> **Why `roles/bigquery.readSessionUser` on the trainer, and the class of bug it represents.** BigQuery splits reading into **three** permissions that people routinely assume are one: `jobUser` *runs a query*, `dataViewer`/`dataEditor` *reads table data over the REST API*, and `readSessionUser` *reads table data over the high-throughput gRPC **Storage Read API***. Any pandas-shaped call — `.to_dataframe()`, which `train.py` uses on line 26 — **silently upgrades itself to the third path** whenever `google-cloud-bigquery-storage` is installed, which it is, in both the venv and the training image. Omit the role and Vertex AI's custom job dies with `PermissionDenied: 403 ... the user does not have 'bigquery.readsessions.create' permission`.
>
> The reason this is worth a box rather than a line: **the identical script passes on your laptop.** Locally you run as *yourself*, and you are Owner on your own project, so the Storage Read path is silently permitted. The first time the code executes under a least-privilege service account — which is the *entire point* of the trainer SA — it fails. That gap between "works as me" and "works as the robot" is where a large share of real GCP incidents live, and being able to name it matters more than the fix itself.

> **Why `roles/logging.logWriter` on the trainer — the role that hides all the other bugs.** A container running under a service account can only write to Cloud Logging if that SA is allowed to. Omit this role and the container's `stdout` and `stderr` are **silently discarded**: not buffered, not delayed, not filed somewhere obscure — *gone*. Vertex still reports the exit code, so what you see is:
>
> ```
> The replica workerpool0-0 exited with a non-zero status of 1.
> To find out more about why your job exited please check the logs: <link>
> ```
>
> …and the link takes you to an empty log. Every `gcloud logging read` you try returns the same four service-level lines (`Waiting for job to be provisioned`, `Job is running`, the exit notice, `Job failed`) and none of your own. You then spend an hour debugging your *log query*, adjusting `--freshness`, switching `textPayload` for `jsonPayload.message`, widening the severity filter — all of it wasted, because there is nothing there to find.
>
> This role grants **no access to any data**. It is pure observability, and its absence converts every future failure in this project into a guessing game. It bites at Step 15.3 rather than Step 7.4 because `CustomContainerTrainingJob` logs through Vertex's own service identity, while a **Pipeline task's container logs as the SA you gave it**. Grant it now.
>
> **A pattern worth naming:** this is the third distinct failure in this build caused by a missing role that nobody thinks of as a "permission" — `readSessionUser` (above), `pubsub.viewer` (below), and now `logWriter`. GCP's default-deny model means *observability is itself a privilege you must grant*. That sentence is the whole lesson.

> **Why `roles/pubsub.viewer` on the Dataflow SA and not just `subscriber`.** `subscriber` lets the worker *pull messages*; it does **not** let it *read the subscription's configuration*. Dataflow's service reads that config at launch to check the ack deadline and reject unsupported settings. Without `viewer`, every launch emits `JOB_MESSAGE_WARNING: Querying the configuration of Pub/Sub subscription ... failed ... PERMISSION_DENIED ... GETTING_PUBSUB_SUBSCRIPTION_FAILED`. The job still runs — but you now have a permanent red herring at the top of every log, and when a *real* failure appears below it you will waste twenty minutes chasing the wrong line. Grant the role; the warning disappears.

> **HCL gotcha (recurring across all three projects):** every attribute in a nested block goes on its **own line**. Collapsing something like `time_partitioning { type = "DAY" field = "event_time" }` onto one line is a parse error. The `schema = jsonencode([...])` lists above are safe because they're JSON strings, not HCL blocks — but `time_partitioning` is a real HCL block and must stay multiline.

```bash
terraform init && terraform plan && terraform apply
terraform state list
```

> **Gotcha — the first `terraform apply` can fail on API enablement.** You may see `Error 403: Cloud Pub/Sub API has not been used in project reco-engine-adv before or it is disabled`. The `depends_on` lines above order the graph correctly, but API enablement is **eventually consistent** — the API is "on" a few seconds before Google's backends agree. The fix is not a code change: wait ~30 seconds and run `terraform apply` again. It will converge.

**What is deliberately *not* in this file:** `rank_training` (Phase 10), the `ranker` BQML model (Phase 10), `model_health` and `feature_drift` (Phase 14), the `*_latest` views (Phase 4), and every dbt model (Phase 16). Those are created by SQL. Declaring a table in both Terraform *and* SQL makes the two fight over its schema — so each object has exactly one owner. `delete_contents_on_destroy = true` on the dataset is what lets `terraform destroy` clean up the SQL-created objects anyway.

✅ **Checkpoint 1:** `terraform state list` shows the topic + subscription, **both buckets**, **eight** BigQuery tables, and three service accounts. Confirm the locations are genuinely different — this is the one thing Phase 8 will not forgive:

```bash
gcloud storage buckets describe gs://reco-engine-adv-artifacts --format='value(location)'   # → EU
gcloud storage buckets describe gs://reco-engine-adv-vectors   --format='value(location)'   # → EUROPE-WEST3
```

If the second one prints `EU`, you copied the wrong `location` line. **A bucket's location is immutable** — you cannot fix this in place. Delete it, correct the HCL, re-apply.

> **No extra IAM needed.** Both service accounts that touch GCS (`trainer`, `df-worker`) already hold `roles/storage.objectAdmin` at the **project** level, which covers any bucket in the project — including one that didn't exist when the role was granted.

---

# Phase 2 — Simulate users with learnable preferences (~1.5 hrs)

**Goal:** An interaction stream with real structure — each user has a hidden preferred category and mostly interacts with it. That hidden preference is (a) the signal the two-tower model must learn, and (b) your ground truth for measuring whether it did.

> **Term — Pub/Sub:** a managed message queue. Producers *publish* to a **topic**; consumers *subscribe* and process independently. It decouples "something happened" from "something processed it," so a traffic spike queues up instead of knocking your pipeline over.

```bash
mkdir -p ~/dev/reco-engine-adv/generator
```

## Step 2.1 — The generator

**`generator/generate.py`**:

```python
import os
import json
import time
import random
import uuid
from datetime import datetime, timezone

from google.cloud import pubsub_v1

PROJECT = os.environ["PROJECT"]
pub = pubsub_v1.PublisherClient()
TOPIC = pub.topic_path(PROJECT, "user-events")

CATEGORIES = ["electronics", "books", "home", "sports", "beauty"]
N_ITEMS, N_USERS = 500, 300
SLEEP = float(os.environ.get("SLEEP", "0.1"))

# Seeded BEFORE the catalog/users are drawn: re-running this script always
# rebuilds the IDENTICAL catalog and the identical hidden preferences, so the
# ground-truth files you .gitignore can always be regenerated exactly.
random.seed(7)
ITEMS = {f"i{n:04d}": {"category": random.choice(CATEGORIES),
                       "price": round(random.uniform(5, 200), 2)}
         for n in range(N_ITEMS)}
USERS = {f"u{n:04d}": random.choice(CATEGORIES) for n in range(N_USERS)}

# Dump ground truth ONCE. Step 2.2 loads these into BigQuery; the retrieval
# test (Phase 9) and the simulator (Phase 12) read them.
if not os.path.exists("catalog.jsonl"):
    with open("catalog.jsonl", "w") as f:
        for iid, it in ITEMS.items():
            f.write(json.dumps({"item_id": iid, **it}) + "\n")
    with open("users.jsonl", "w") as f:
        for uid, pref in USERS.items():
            f.write(json.dumps({"user_id": uid, "preferred_category": pref}) + "\n")
    print("wrote catalog.jsonl + users.jsonl")

# Weighted so ~57% of events are click/add_to_cart/purchase — the "positive"
# interactions the two-tower model trains on.
EVENT_TYPES = ["view", "view", "view", "click", "click", "add_to_cart", "purchase"]
BY_CAT = {c: [i for i, it in ITEMS.items() if it["category"] == c] for c in CATEGORIES}
ALL_ITEMS = list(ITEMS)


def sample_item(pref):
    # 70% of interactions land in the user's preferred category — THE SIGNAL.
    return random.choice(BY_CAT[pref]) if random.random() < 0.7 else random.choice(ALL_ITEMS)


print(f"publishing to {TOPIC} — Ctrl-C to stop")
while True:
    uid = random.choice(list(USERS))
    ev = {
        "event_id": str(uuid.uuid4()),
        "user_id": uid,
        "item_id": sample_item(USERS[uid]),
        "event_type": random.choice(EVENT_TYPES),
        "event_time": datetime.now(timezone.utc).isoformat(),
    }
    pub.publish(TOPIC, json.dumps(ev).encode("utf-8"))
    time.sleep(SLEEP)
```

**`generator/requirements.txt`**:

```
google-cloud-pubsub==2.23.0
```

Run it **in its own terminal** and leave it running while you build:

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate
pip install -r generator/requirements.txt
cd generator
PROJECT=reco-engine-adv SLEEP=0.1 python generate.py
```

At `SLEEP=0.1` that's ~10 events/sec ≈ **36,000 events/hour**. About 57 % of those are the `click`/`add_to_cart`/`purchase` events the two-tower model actually trains on, so **50k events ≈ 28k training pairs**. `train.py` refuses to run below **5,000 pairs** and you want **20k+** for a model worth showing — so leave this running for roughly **90 minutes** in the background while you do Phases 3–6. Ctrl-C stops it; re-running appends more events to the same catalog.

> **A second reason to let it run: item coverage.** `train.py` builds its item vocabulary from items that were *actually interacted with*. An item nobody has ever clicked has **no embedding**, so it never enters the Vector Search index and can never be retrieved — a real cold-start property of every two-tower recommender, not a bug. At 50k events essentially all 500 items get hit; at 5k events a handful won't. Everything downstream of training therefore intersects with the items that actually have embeddings, rather than assuming the full catalogue is there.

## Step 2.2 — Verify the topic is actually receiving messages

In a **second** terminal (`pull` without `--auto-ack` peeks at a message without consuming it):

```bash
gcloud pubsub subscriptions pull events-df-sub --limit=1 --format=json
```

You should see one JSON event body. (If you get nothing, the generator isn't publishing — check its terminal for an auth error.)

## Step 2.3 — Load the ground truth into BigQuery (one-time)

```bash
cd ~/dev/reco-engine-adv/generator

bq --project_id=reco-engine-adv load --replace \
  --source_format=NEWLINE_DELIMITED_JSON reco.catalog catalog.jsonl
bq --project_id=reco-engine-adv load --replace \
  --source_format=NEWLINE_DELIMITED_JSON reco.users_truth users.jsonl

bq query --use_legacy_sql=false 'SELECT COUNT(*) AS items FROM reco.catalog'        # 500
bq query --use_legacy_sql=false 'SELECT COUNT(*) AS users FROM reco.users_truth'    # 300
```

> **⚠️ `--replace` is not optional.** `bq load` **appends** by default. Run Step 2.3 twice without `--replace` and you get 1,000 catalog rows — two copies of every item. Every downstream join then fans out: `item_features` doubles, the Phase 10 training set doubles, `popular_items()` returns the same item twice, and nothing errors out. It just quietly produces wrong numbers. `--replace` truncates first, so the step is idempotent no matter how many times you run it.

✅ **Checkpoint 2:** the `pull` command returns a real event, and `catalog` has exactly **500** rows / `users_truth` exactly **300**.

---

# Phase 3 — Stream events into BigQuery with Dataflow (~2 hrs)

**Goal:** Land raw events in `reco.events` continuously via Apache Beam on Dataflow.

> **Term — Apache Beam / Dataflow:** Beam is a portable pipeline SDK (you describe a dataflow graph: read → transform → write); **Dataflow** is Google's managed runner that executes it on autoscaling workers. A Pub/Sub→BigQuery subscription could do this raw hop with zero code — but Beam is the transferable skill, and it's where you'd add sessionization, windowing, or enrichment later.

```bash
mkdir -p ~/dev/reco-engine-adv/dataflow
```

## Step 3.1 — The pipeline

**`dataflow/stream_events.py`**:

```python
import json
import argparse

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

SCHEMA = ("event_id:STRING,user_id:STRING,item_id:STRING,"
          "event_type:STRING,event_time:TIMESTAMP")


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)    # NOT --project (Beam owns that flag)
    ap.add_argument("--subscription", required=True)
    args, beam_args = ap.parse_known_args()

    opts = PipelineOptions(
        beam_args,
        streaming=True,
        save_main_session=True,    # ships the top-level `json` import to the workers
        project=args.project_id,
    )

    with beam.Pipeline(options=opts) as p:
        (p
         | "Read" >> beam.io.ReadFromPubSub(subscription=args.subscription)
         | "Parse" >> beam.Map(lambda m: json.loads(m.decode("utf-8")))
         | "ToBQ" >> beam.io.WriteToBigQuery(
             f"{args.project_id}:reco.events",
             schema=SCHEMA,
             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
             create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER))


if __name__ == "__main__":
    run()
```

**`dataflow/requirements.txt`**:

```
apache-beam[gcp]==2.75.0
```

> **⚠️ Why `2.75.0` and not an older pin.** Dataflow ages Apache Beam SDKs out: **Beam 2.60 was marked *Deprecated* on 17 Oct 2025**, and versions **2.60.x–2.62.x carry a documented streaming bug** (`KeyCommitTooLargeException` on Streaming Appliance jobs — Google's own guidance is "upgrade to 2.63.0 or later"). Submitting a job from a deprecated SDK gets you a warning today and a *disrupted, throttled* streaming job later. 2.75.0 is supported into mid-2027. Before you build, glance at [Dataflow's SDK version support status](https://cloud.google.com/dataflow/docs/support/sdk-version-support-status) and take the newest **Supported** row.

> **The three Beam gotchas — all three will bite you, in this order:**
> 1. **`--project_id`, never `--project`.** Beam claims `--project` for itself. If your `argparse` consumes it, the Dataflow launch dies with `ValueError: Invalid GCS path` or an opaque options error, because Beam never sees the project it needs to stage into.
> 2. **`save_main_session=True`.** Without it the pipeline runs fine locally on DirectRunner and then fails **on Dataflow only** with `NameError: name 'json' is not defined` — the workers get your lambda but not your module-level imports.
> 3. **Read the declared subscription, not the topic.** `ReadFromPubSub(topic=...)` makes Beam create its own subscription at runtime, which requires `roles/pubsub.editor` on the worker SA. Reading `events-df-sub` (declared in Phase 1) keeps the SA down to `pubsub.subscriber` + `pubsub.viewer` — pull the messages, read the subscription's config, create nothing. (`viewer` is the one added in Phase 1 to silence the `GETTING_PUBSUB_SUBSCRIPTION_FAILED` warning; it is still a long way short of `editor`.)

## Step 3.2 — Local smoke test (DirectRunner, free)

```bash
cd ~/dev/reco-engine-adv
source .venv-beam/bin/activate           # the Beam-only venv
pip install -r dataflow/requirements.txt

python dataflow/stream_events.py \
  --project_id reco-engine-adv \
  --subscription projects/reco-engine-adv/subscriptions/events-df-sub \
  --runner DirectRunner
```

Let it run for ~30 seconds, then Ctrl-C. In another terminal:

```bash
bq query --use_legacy_sql=false 'SELECT COUNT(*) AS n FROM reco.events'
```

Non-zero → the pipeline, the schema, and the permissions all work. (The local run **acks** the messages it reads, so those few hundred events won't be re-delivered to the Dataflow job. Harmless — the generator is still producing.)

## Step 3.3 — Launch on Dataflow

```bash
python dataflow/stream_events.py \
  --project_id reco-engine-adv \
  --subscription projects/reco-engine-adv/subscriptions/events-df-sub \
  --runner DataflowRunner \
  --project reco-engine-adv \
  --region europe-west3 \
  --temp_location gs://reco-engine-adv-artifacts/dataflow/temp \
  --staging_location gs://reco-engine-adv-artifacts/dataflow/staging \
  --service_account_email df-worker@reco-engine-adv.iam.gserviceaccount.com \
  --enable_streaming_engine \
  --machine_type e2-standard-2 \
  --num_workers 1 \
  --max_num_workers 2 \
  --job_name reco-stream-events
```

Note that `--project` appears here **as a Beam flag** (after `--project_id`, which is yours). That's the whole point of gotcha #1.

**Why the three extra flags** — they are not decoration, they are your defence against the failure in the next box:

| Flag | What it does |
|------|--------------|
| `--enable_streaming_engine` | Moves shuffle and state off the VM and into Google's managed backend. The worker becomes a thin executor, so a **small** machine is enough — and small machines are the ones that are actually available. |
| `--machine_type e2-standard-2` | The default streaming worker is `n1-standard-4`. `e2` is a different, far larger capacity pool than `n1`, and 2 vCPU is a smaller ask than 4. Both facts raise your odds of getting scheduled. |
| `--num_workers 1` | Ask for exactly one VM. You are streaming a few hundred events/min from a laptop generator; one is plenty, and it's one allocation that has to succeed instead of two. |

> **⚠️ `ZONE_RESOURCE_POOL_EXHAUSTED` — the failure that isn't your fault.** The most likely way this step dies is:
>
> ```
> JOB_MESSAGE_ERROR: Startup of the worker pool in europe-west3 failed to bring up any of the
> desired 1 workers. This is likely a quota issue or a Compute Engine stockout. ...
> ZONE_RESOURCE_POOL_EXHAUSTED: Instance '...-harness-lv7w' creation failed: The zone
> 'projects/reco-engine-adv/zones/europe-west3-c' does not have enough resources available to
> fulfill the request. Try a different zone, or try again later.
> ... Workflow failed.
> ```
> Dataflow retries the VM creation for ~5 minutes, fails ten times in a row, then gives up and your local `beam.Pipeline` context manager raises `DataflowRuntimeException: ... State: FAILED`.
>
> **Read what it says.** This is *not* a quota error (a quota error names the quota and says `QUOTA_EXCEEDED`), and it is *not* an IAM error. It is a **stockout**: Google physically has no free VM of that shape in that zone right now. `europe-west3` (Frankfurt) is a small, chronically tight region — it is the correct region for this project because the BigQuery dataset is EU and the users are European, but it is the region most likely to be out of machines at 3 a.m. Nothing in your code is broken. Re-running the *identical* command an hour later often just works, which is exactly what makes this bug so disorienting the first time.
>
> **The escape ladder — climb it in order, stop at the first rung that launches:**
>
> 1. **Add the three flags above** (`--enable_streaming_engine --machine_type e2-standard-2 --num_workers 1`) if you haven't. Smaller, more-common machine, single instance. This alone clears most stockouts.
> 2. **Pin a different zone.** Dataflow picked `europe-west3-c` for you; you can override it. Frankfurt has `-a`, `-b`, `-c`, and they are independently stocked:
>    ```bash
>    # ... same flags as above, plus:
>      --worker_zone europe-west3-a
>    ```
>    If `-a` is also exhausted, try `-b`. (The flag is `--worker_zone`, singular. The old `--zone` is deprecated and silently ignored by newer SDKs.)
> 3. **Move the job to a bigger region.** Your artifacts bucket is `location = "EU"` — a **multi-region** bucket — and your BigQuery dataset is `EU` too, so the job can run from any European region with no cross-region cost and no code change. `europe-west1` (Belgium) and `europe-west4` (Netherlands) are Google's large, well-stocked European regions:
>    ```bash
>      --region europe-west1
>    ```
>    Drop `--worker_zone` when you change region — a zone from the old region is an instant error.
> 4. **Wait.** Stockouts clear. If all of the above fail, the region is genuinely full; come back in an hour.
>
> **The one thing you must remember afterwards:** whatever region the job actually launched in is the region you must use to *stop* it. `gcloud dataflow jobs list --region europe-west3` will cheerfully report "no jobs" while a job you launched in `europe-west1` bills away in the background. Off-switches are region-scoped. Write down the region you used.
>
> **The one-line version of this:** *"Streaming jobs in small regions hit capacity stockouts. I made the worker request smaller — Streaming Engine plus a 2-vCPU e2 instead of the default 4-vCPU n1 — so the scheduler could actually place it, and I kept the fallback region in the runbook."* That is a real production instinct, and it is the difference between "the cloud is broken" and "I sized my request to the capacity that exists."

> **What "stuck" looks like (it isn't).** The launch command returns quickly, but the job spends **3–6 minutes** provisioning a worker and pulling the container before a single row lands in BigQuery. `SELECT COUNT(*)` staying flat for five minutes is normal. Watch the job in **Dataflow → Jobs** in the console; it turns from *Starting* to *Running*. What you should **not** see is the same `JOB_MESSAGE_ERROR` line repeating every 30 seconds — that's the stockout above, and no amount of waiting inside that job will fix it.

> **⚠️ Your terminal will now hang forever, and that is correct.** The pipeline uses `with beam.Pipeline(...) as p:`, and on exit that context manager calls `wait_until_finish()`. A **streaming** job never finishes — by definition. So your local Python process sits there politely waiting until the heat death of the universe. It is a foreground log tailer at this point, nothing more.
>
> **Ctrl-C it. This does NOT stop the Dataflow job.** The job was submitted to Google's servers minutes ago and runs there, entirely independent of your laptop. Ctrl-C only stops your local process from polling for status; you'll get a `KeyboardInterrupt` out of `wait_until_finish()` and that's expected. Confirm with `gcloud dataflow jobs list --region europe-west3 --status active` — still `Running`.
>
> This trips people up constantly: they Ctrl-C, assume they've stopped everything, close the laptop, and get a bill. **The only thing that stops the job is the `drain` command in Step 3.4.**

## Step 3.4 — Verify, then know how to stop it

```bash
# rows should climb every time you re-run this while the generator is up:
bq query --use_legacy_sql=false 'SELECT COUNT(*) AS n FROM reco.events'

# the off-switch — DO THIS AT THE END OF EVERY SESSION:
gcloud dataflow jobs list --region europe-west3 --status active
gcloud dataflow jobs drain <JOB_ID> --region europe-west3
```

> **Drain, don't cancel.** `drain` stops pulling new messages and lets in-flight ones finish writing; `cancel` drops them. Either way the job stops billing — this is off-switch #1 of the three from Phase 0.

> **`--region` must match where the job actually launched.** If you climbed to rung 3 of the stockout ladder and launched in `europe-west1`, then `--region europe-west1` here too. `jobs list` on the wrong region returns an empty list, which reads exactly like "nothing is running" and is the single easiest way to leave a streaming job billing overnight.

✅ **Checkpoint 3:** `SELECT COUNT(*) FROM reco.events` grows while the generator and the Dataflow job both run.

---

# Phase 4 — Feature engineering in BigQuery (~2 hrs)

**Goal:** Build `user_features` and `item_features` as **append-only snapshots** from the event log + catalog. These tables *are* the offline feature store.

> **Term — feature store:** a system that computes features **once**, stores them with a timestamp, and serves the *same* values to training and to serving. The problem it solves has a name — **training-serving skew**: your training pipeline computes `events_7d` one way (in SQL, over history) and your API computes it another way (in Python, at request time), the two silently disagree, and your model degrades in production for no visible reason. One definition, two consumers.
>
> **Term — point-in-time join:** for *training*, each historical example should join to the feature values **as they were at that moment**, not today's. Otherwise the future leaks into the past and your offline metrics look magnificent and mean nothing. The `feature_timestamp` column is what makes this possible, and it's the core reason feature stores exist.

## Step 4.1 — User features (append a snapshot)

Run in the **BigQuery query editor**:

```sql
INSERT INTO `reco-engine-adv.reco.user_features`
  (user_id, feature_timestamp, events_7d, purchases_30d, distinct_items_7d, top_category)
WITH ev AS (
  -- top_category needs the CATEGORY, but the event stream only carries item_ids
  SELECT e.user_id, e.item_id, e.event_type, e.event_time, c.category
  FROM `reco-engine-adv.reco.events` e
  JOIN `reco-engine-adv.reco.catalog` c USING (item_id)
)
SELECT
  user_id,
  CURRENT_TIMESTAMP() AS feature_timestamp,
  COUNTIF(event_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)) AS events_7d,
  COUNTIF(event_type = 'purchase'
          AND event_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)) AS purchases_30d,
  COUNT(DISTINCT IF(event_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY),
                    item_id, NULL)) AS distinct_items_7d,
  APPROX_TOP_COUNT(category, 1)[OFFSET(0)].value AS top_category
FROM ev
GROUP BY user_id;
```

## Step 4.2 — Item features (append a snapshot)

```sql
INSERT INTO `reco-engine-adv.reco.item_features`
  (item_id, feature_timestamp, category, price, popularity_7d)
SELECT
  c.item_id,
  CURRENT_TIMESTAMP() AS feature_timestamp,
  c.category,
  c.price,
  COUNTIF(e.event_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)) AS popularity_7d
FROM `reco-engine-adv.reco.catalog` c
LEFT JOIN `reco-engine-adv.reco.events` e USING (item_id)
GROUP BY c.item_id, c.category, c.price;
```

The `LEFT JOIN` matters: an item nobody has touched still gets a row, with `popularity_7d = 0`. An `INNER JOIN` would silently drop cold items out of the catalog — and out of your ranker's world.

> **⚠️ Why `INSERT`, and not `CREATE OR REPLACE TABLE`.** Two reasons, and both are load-bearing:
> 1. **Terraform owns these tables** (Phase 1). `CREATE OR REPLACE TABLE` drops and recreates them, so Terraform and SQL end up fighting over the schema. `INSERT` writes rows into a table that already exists — no conflict, ever.
> 2. **A feature store is a history, not a snapshot.** One row per entity per `feature_timestamp` is what makes point-in-time joins, drift detection (Phase 14), and the Vertex Feature Registry (Phase 5) actually work. Replacing the table on every run throws that history away — and would make the `*_latest` views below pointless dead code.
>
> The price of append-only is the fan-out trap in the next box. Pay it; it's worth it.

## Step 4.3 — The `*_latest` views (read these, never the raw tables)

```sql
CREATE OR REPLACE VIEW `reco-engine-adv.reco.user_features_latest` AS
SELECT * EXCEPT(rn) FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY feature_timestamp DESC) AS rn
  FROM `reco-engine-adv.reco.user_features`
) WHERE rn = 1;

CREATE OR REPLACE VIEW `reco-engine-adv.reco.item_features_latest` AS
SELECT * EXCEPT(rn) FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY feature_timestamp DESC) AS rn
  FROM `reco-engine-adv.reco.item_features`
) WHERE rn = 1;

CREATE OR REPLACE VIEW `reco-engine-adv.reco.user_embeddings_latest` AS
SELECT * EXCEPT(rn) FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY trained_at DESC) AS rn
  FROM `reco-engine-adv.reco.user_embeddings`
) WHERE rn = 1;
```

(The third view is empty until Phase 7 trains a model — the *table* exists from Terraform, so the view compiles fine now.)

> **⚠️ The fan-out trap — the bug this whole design creates, and how to never hit it.** After the second snapshot, `user_features` holds **two rows per user**. Any query that joins the **raw** table instead of the **`_latest` view** silently multiplies its result set: your Phase 10 training rows double, `popular_items()` returns the same item three times, your CTRs go strange, and nothing errors. **Rule for the rest of this guide: every read of a feature table goes through `_latest`.** The only exceptions — the two places that *want* the full history — are the Feature Registry source (Phase 5) and the drift query (Phase 14), and both are called out where they appear.

## Step 4.4 — Sanity check: is the signal actually in the features?

```sql
SELECT uf.top_category, ut.preferred_category, COUNT(*) AS n
FROM `reco-engine-adv.reco.user_features_latest` uf
JOIN `reco-engine-adv.reco.users_truth` ut USING (user_id)
GROUP BY 1, 2 ORDER BY n DESC LIMIT 10;
```

The top rows should be the **diagonal** — `top_category = preferred_category`. That's the 70 % preference showing up in aggregate. If it isn't there, your model has nothing to learn and every later phase will look broken for the wrong reason.

## Step 4.5 — Schedule both inserts hourly

Console → **BigQuery → Scheduled queries → Create scheduled query**. Paste the Step 4.1 SQL, repeat **every 1 hour**, location **EU**, name it `user_features_hourly`. Repeat for Step 4.2 as `item_features_hourly`.

> **Gotcha:** scheduled queries run on the **BigQuery Data Transfer API**. If the console refuses with `BigQuery Data Transfer API has not been used in project…`, it's because that API isn't on — it's in the Phase 1 Terraform (`bigquerydatatransfer.googleapis.com`), so `terraform apply` and retry. You need at least **two** snapshots before Phase 14's drift query returns anything, so let this run for a couple of hours (or just execute Step 4.1 by hand a second time).

✅ **Checkpoint 4:** both feature tables have rows, the three `_latest` views return exactly one row per entity, and `top_category` matches ground-truth preference for most users.

---

# Phase 5 — Register features in Vertex AI Feature Store (~1.5 hrs)

**Goal:** Catalog the two BigQuery tables in the **Feature Registry** — the current (non-legacy) Feature Store design. No data moves. BigQuery *is* the offline store; the registry adds lineage, discovery, governance, and the hook for online serving.

```bash
mkdir -p ~/dev/reco-engine-adv/featurestore
cd ~/dev/reco-engine-adv && source .venv/bin/activate
pip install "google-cloud-aiplatform>=1.71"
```

**`featurestore/register.py`**:

```python
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
```

```bash
python featurestore/register.py
```

> **Gotcha — the `feature_timestamp` column name is not a suggestion.** The Feature Registry requires the source table to carry a `TIMESTAMP` column literally named **`feature_timestamp`**. Call it `as_of` or `computed_at` and `FeatureGroup.create` rejects the source. That's why the Phase 1 schema and the Phase 4 inserts both spell it exactly that way.

> **Never** use the legacy `aiplatform.Featurestore(...)` / `EntityType` API from older blog posts. It's deprecated, it sunsets in Feb 2027, and it is the single easiest way to end up having cargo-culted a 2022 tutorial. If a code sample you find online creates an "entity type," you're reading the dead API.

✅ **Checkpoint 5:** **Vertex AI → Feature Store → Feature Registry** shows both feature groups with their features, sourced from BigQuery.

---

# Phase 6 — Online serving: BigQuery-direct + a Bigtable demo (~1.5 hrs)

**Goal:** Serve features at request time — the cheap, always-available way for development, plus a *brief*, fully-deleted demonstration of the production path.

## Step 6.1 — The dev path (free): read the `_latest` views

Nothing to build — you made the views in Step 4.3. The API (Phase 11) queries `user_features_latest`, `item_features_latest`, and `user_embeddings_latest` directly. At portfolio QPS that's tens of milliseconds and costs nothing.

That is a legitimate architecture, not a cop-out, and the reason is money:

> **The senior sentence this phase buys you:** *"Bigtable online serving bills per node-hour with no scale-to-zero — hundreds of euros a month at a single node, so for a personal-scale deployment I serve features and batch-computed user vectors straight from BigQuery — and I demonstrated the Bigtable path end-to-end without leaving it idle."* That's platform-economics fluency, not tutorial-following. Say the trade-off out loud before anyone asks.

## Step 6.2 — The Bigtable online store: create → sync → read → **delete**

This script creates the expensive resource, measures it *honestly*, and **deletes it in a `finally` block** so a crash can't leave the meter running.

Three things in it look like over-engineering and are not. Each one is a bug this script hit, in order, on a real run:

1. **The store name has a timestamp in it.** Not cosmetic — see the tombstone gotcha below.
2. **The polling loop prints `str(e)`, not just `type(e).__name__`.** Fourteen identical `NotFound`s with the message stripped off tell you nothing.
3. **The warm-read loop tolerates failures and counts them.** A benchmark that aborts on one transient 404 is a bad benchmark.

**`featurestore/online_store_demo.py`**:

```python
import time

import vertexai
from vertexai.resources.preview import feature_store as fs

PROJECT, REGION = "reco-engine-adv", "europe-west3"

# ── Why the name has a timestamp in it ───────────────────────────────────────
# Deleting a FeatureOnlineStore is a long-running operation. `store.delete()`
# returns as soon as Google ACCEPTS the delete, not when it FINISHES it — and
# the name stays reserved (tombstoned) for many minutes afterwards. Re-running
# with a fixed name then dies on:
#     ALREADY_EXISTS: Re-using the same name as a FeatureOnlineStore being deleted
# Meanwhile `FeatureOnlineStore.list()` returns [] the whole time, because the
# store leaves the LIST when the delete is accepted but the NAME is held until it
# completes. The list is a BILLING check, not a NAME-AVAILABILITY check.
# Don't wait on the name. The name is arbitrary. Make it unique per run.
STORE_NAME = f"reco_online_{int(time.time())}"
VIEW_NAME = "user_features_view"
DEMO_USER = "u0007"

WARM_READS = 20        # how many SUCCESSFUL reads to time
SETTLE_SECS = 20       # let the sync propagate across replicas before benchmarking
MAX_ATTEMPTS = 60      # cap total warm-loop attempts so a broken store can't hang us

vertexai.init(project=PROJECT, location=REGION)

store = None
try:
    print(f"creating the Bigtable online store '{STORE_NAME}' — BILLING STARTS NOW "
          f"(billed per node-hour) ...")
    store = fs.FeatureOnlineStore.create_bigtable_store(STORE_NAME)
    print("online store:", store.resource_name)

    # A FeatureView = "which rows to sync into the online store, and how often".
    # Point it at the _latest VIEW: the online store wants exactly ONE row per
    # entity id, and the raw append-only table has one per snapshot.
    view = store.create_feature_view(
        name=VIEW_NAME,
        source=fs.utils.FeatureViewBigQuerySource(
            uri=f"bq://{PROJECT}.reco.user_features_latest",
            entity_id_columns=["user_id"],
        ),
        sync_config="0 * * * *",          # hourly cron; we trigger one immediately below
    )
    print("feature view:", view.resource_name)

    view.sync()
    print("sync started — first sync usually takes 3–8 minutes ...")

    deadline = time.time() + 900          # give up after 15 minutes
    while time.time() < deadline:
        try:
            t0 = time.perf_counter()
            resp = view.read(key=[DEMO_USER])
            cold_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            # Print the MESSAGE, not just the class name. "NotFound" fourteen times
            # tells you nothing; the message tells you WHICH NotFound it is.
            print("  not synced yet:", type(e).__name__, "-", str(e)[:140])
            time.sleep(30)
            continue

        # ---- First successful read. It is NOT a latency measurement. ----
        # It paid for gRPC channel setup + TLS handshake + client->region RTT.
        # Report it honestly as a cold read, then measure the steady state.
        print(f"\ncold read (channel setup + TLS + first RPC): {cold_ms:.1f} ms")
        print(resp.to_dict())

        # ---- Let the sync settle before benchmarking ----
        # The online store is eventually consistent across serving replicas. The
        # first read can succeed on one replica while the next read lands on
        # another that hasn't got the row yet — a transient 404 seconds after a
        # successful read. A benchmark that aborts on that is a bad benchmark.
        print(f"\nletting the sync settle for {SETTLE_SECS}s before benchmarking ...")
        time.sleep(SETTLE_SECS)

        lat, misses, attempts = [], 0, 0
        while len(lat) < WARM_READS and attempts < MAX_ATTEMPTS:
            attempts += 1
            try:
                t0 = time.perf_counter()
                view.read(key=[DEMO_USER])
                lat.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                # Tolerate and COUNT it. Do not let one transient 404 kill the run.
                misses += 1
                print(f"  transient read failure ({type(e).__name__}) — "
                      f"{misses} so far, retrying")
                time.sleep(2)

        if len(lat) < 3:
            print(f"\nonly {len(lat)} successful reads in {attempts} attempts — "
                  f"the store is not serving reliably. No latency number to report.")
        else:
            lat.sort()
            p50 = lat[len(lat) // 2]
            p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
            print(f"\nWARM READ  median {p50:.1f} ms   p95 {p95:.1f} ms   "
                  f"min {lat[0]:.1f} ms   max {lat[-1]:.1f} ms")
            print(f"           n={len(lat)} successful / {attempts} attempts "
                  f"({misses} transient failures)")
            print("^^^ SCREENSHOT THIS — the warm median is the online-serving number.")
        break
    else:
        print("sync did not finish within 15 min — deleting anyway, cost > curiosity")

except Exception as e:
    # Anything unexpected: say so loudly, but STILL fall through to the delete.
    print(f"\nUNEXPECTED FAILURE: {type(e).__name__}: {e}")
    print("(deleting the store anyway — the meter must stop)")

finally:
    if store is not None:
        print("\ndeleting the online store (stops the meter) ...")
        store.delete(force=True)          # force=True also deletes the child FeatureView
        print("delete accepted — the meter stops now.")
        print("(The delete is a long-running op, so the NAME stays reserved a few more")
        print(" minutes. That's why this script picks a fresh name every run.)")
```

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate
python featurestore/online_store_demo.py
```

Expect: a few minutes of `not synced yet`, then a **cold read of ~150–250 ms**, then a settle pause, then a `WARM READ` block reporting median / p95 / min / max over 20 reads. Screenshot that block — it's the artifact this phase exists to produce.

**What the warm median actually measures, and why it isn't single-digit.** Run from a laptop, this is a round trip over the public internet into `europe-west3` plus the Feature Store's serving API, and *then* a Bigtable lookup. On the run this guide was written against it came out at **median 102.8 ms** (p95 191.5, min 83.2, max 191.5, n = 20/20). Bigtable's own contribution to that is a small fraction of it. The number is honest and it is still the right number to report — provided you say **where you measured it from**. A client inside the same region talks to a different network; a client on your sofa does not. What the phase proves is not "Bigtable is fast," it is *what an online store costs and what it takes to reach one* — and those are the two facts the Phase 6.1 decision turns on.

> **⚠️ Do not report the cold read as your latency — and the naive version of this script invites you to.** The obvious way to write it is: poll until a read succeeds, time that read, `break`, screenshot. That number will be **~200 ms**, and it is *not* your online-serving latency. It is one read that also paid for gRPC channel setup, a TLS handshake, and a client→region round trip. Report *that* as "Bigtable online feature serving latency" and the first person who asks "over how many reads?" gets the answer "one," and the number evaporates — taking the whole cost/latency argument with it, because you have just accidentally made the case that Bigtable is **slow *and* expensive**.
>
> The honest artifact is the **pair**, with the measurement point named: *"Cold read 216 ms, warm median 103 ms over 20 reads, measured from a laptop outside GCP — the cold read's extra ~110 ms is channel establishment and TLS, not the store."* That is a stronger sentence than either number alone, and unlike a bare median it survives the first follow-up question.

> **⚠️ Gotcha — `ALREADY_EXISTS: Re-using the same name as a FeatureOnlineStore being deleted`.** You will hit this the *second* time you run the script, and it is the most confusing error in the phase, because the check you'd naturally run to rule it out **says you're fine**:
>
> ```
> $ python -c "... print(fs.FeatureOnlineStore.list())"
> []
> ```
>
> Empty. No store. And yet `create_bigtable_store("reco_online")` still 409s.
>
> **Both facts are true, and they answer different questions.** `store.delete()` is a *long-running operation*: it returns the moment Google **accepts** the delete, and the store vanishes from `list()` immediately — but the **name stays reserved** until the delete actually *completes*, which takes several more minutes. So:
>
> | Check | Answers | Does NOT answer |
> |---|---|---|
> | `FeatureOnlineStore.list()` → `[]` | *"Am I being billed?"* — **No.** | *"Is the name free?"* |
> | `create_bigtable_store(name)` → 409 | *"Is the name free?"* — **No.** | *"Am I being billed?"* |
>
> Confusing a billing check for a name-availability check will cost you twenty minutes of staring at an empty list.
>
> **The fix is not to wait.** The name is arbitrary; nothing downstream depends on it; the script deletes whatever it creates. So `STORE_NAME = f"reco_online_{int(time.time())}"` — a fresh name every run, and the tombstone can hold the old one as long as it likes. The teardown that protects your wallet is the same teardown that blocks your next run; sidestep it rather than fight it.

> **⚠️ Gotcha — a `NotFound` *seconds after* a successful read.** The first version of the warm loop assumed that once one read works, all reads work. That is false immediately after a fresh sync. The online store's serving layer is **eventually consistent across replicas**: your cold read can land on a replica that has `u0007` while the very next read lands on one that doesn't yet — a transient `404 Requested entity was not found` a second after a clean success. If that exception escapes, the script dies mid-benchmark (the `finally` block still deletes the store, so your wallet is safe — but the traceback appears *after* the deletion messages because tracebacks go to stderr and prints go to stdout, and the interleaving makes it look like the delete caused the failure. It didn't.).
>
> Hence `SETTLE_SECS` and the tolerant loop. **Keep the attempt count in your screenshot either way.** `n=20 successful / 20 attempts (0 transient failures)` is a claim about your benchmark; `n=20 / 22 (2 transient failures)` is a *more interesting* one — "two of twenty-two reads 404'd within a minute of a fresh sync, because the serving layer is eventually consistent" tells the reader you know what you measured. A median with no denominator quietly drops the inconvenient reads and says nothing at all.

> **Gotcha — if the sync rejects a view as a source.** Some SDK/API versions want a *table*, not a view, behind a `FeatureViewBigQuerySource`. If `create_feature_view` or the sync errors on the source, materialize it once and point at the table instead:
> ```sql
> CREATE OR REPLACE TABLE `reco-engine-adv.reco.user_features_online` AS
> SELECT * FROM `reco-engine-adv.reco.user_features_latest`;
> ```
> then change the `uri` to `bq://reco-engine-adv.reco.user_features_online`.

> **⚠️ How to actually verify nothing is billing — and the two commands that will lie to you.**
>
> **`gcloud bigtable instances list` will show an empty list while the online store is happily billing you.** The Bigtable behind a Vertex `FeatureOnlineStore` is **managed by Vertex inside Google's project, not yours**. Believing that empty list is how you get a three-figure surprise.
>
> **`gcloud ai feature-online-stores list` does not exist in many gcloud versions.** You'll get `ERROR: (gcloud.ai) Invalid choice: 'feature-online-stores'` — which is not "nothing is running," it's "I can't tell you." A command that errors is not a green light.
>
> The check that works everywhere is the Python one. Use this as your off-switch verification:
> ```bash
> python -c "import vertexai; vertexai.init(project='reco-engine-adv', location='europe-west3'); from vertexai.resources.preview import feature_store as fs; print(fs.FeatureOnlineStore.list())"
> ```
> It must print `[]`. (If your gcloud *is* new enough, `gcloud ai feature-online-stores list --region=europe-west3 --project=reco-engine-adv` works too, and the console path is **Vertex AI → Feature Store → Online store**.)

✅ **Checkpoint 6:** the `_latest` views return one row per entity; you have the **warm** latency screenshot (median + p95 + the transient-failure count, *not* the cold read); and `FeatureOnlineStore.list()` prints `[]`.

---

# Phase 7 — Train the two-tower retrieval model (~4 hrs)

**Goal:** The heart of the recommender. One training run produces **item embeddings → GCS** (for Vector Search) and **user embeddings → BigQuery** (for the API) — the entire serving stack is fed from this one script.

> **Term — two-tower model:** two small neural nets — a **user tower** and an **item tower** — trained jointly so that a user's vector ends up close to the vectors of items they engaged with, and far from everything else (using the other items in the same batch as negatives — "in-batch negatives"). At serving time you precompute every item vector once, index them, and recommending becomes: *look up the user's vector, find its nearest item vectors.* A 500-item catalog search becomes one nearest-neighbour query.
>
> **Design choice — batch user embeddings.** The API could embed users *live* (deploy the user tower to an endpoint: extra cost, extra latency, fresher) or read **precomputed vectors from BigQuery** (cheap, simple, refreshed each retrain). This guide does the latter — a very common production pattern — and names the real-time user tower as the explicit upgrade path. Say that trade-off out loud too.

## Step 7.1 — Setup (the Keras 3 landmine)

```bash
mkdir -p ~/dev/reco-engine-adv/training
cd ~/dev/reco-engine-adv && source .venv/bin/activate
```

**`training/requirements.txt`** — note what is **not** in here:

```
tensorflow==2.17.0
tf-keras==2.17.0
tensorflow-io-gcs-filesystem==0.37.1
google-cloud-bigquery==3.25.0
google-cloud-bigquery-storage==2.27.0
google-cloud-storage==2.18.2
db-dtypes==1.3.0
pandas==2.2.2
protobuf==4.25.5
numpy==1.26.4
scipy==1.13.1
```

Install it in **two commands**, and the second one is not optional:

```bash
pip install -r training/requirements.txt
pip install --no-deps tensorflow-recommenders==0.7.3      # --no-deps is load-bearing — read the box
```

> **⚠️ The macOS trap that eats an evening: `tensorflow-recommenders` will try to install a *second* TensorFlow.** TFRS declares its dependency as `tensorflow-macos>=2.9.0 ; sys_platform == "darwin"` — and **`tensorflow-macos` is a dead package whose last release was 2.16.2**. On your Apple-Silicon Mac, a plain `pip install tensorflow-recommenders` therefore drags an *older, different* TensorFlow in on top of the 2.17.0 you just pinned, the two overwrite each other's files in `site-packages`, and you get import errors that point at nothing real.
>
> `--no-deps` is the fix, and it is safe: TFRS's only actual runtime dependencies are TensorFlow and `tf-keras`, and you have already installed both, at the versions you chose. Verify with `pip list | grep -i tensorflow` — you should see `tensorflow 2.17.0`, `tf_keras 2.17.0`, `tensorflow-recommenders 0.7.3`, and **no `tensorflow-macos`**. (On Linux — the Dockerfile in Step 7.4 — the marker doesn't fire, but the same `--no-deps` line is used there so the two environments stay identical.)

> **⚠️ `tensorflow-io-gcs-filesystem` is pinned on purpose, and it is *not* optional on Python 3.12.** TensorFlow 2.17 declares that dependency as `tensorflow-io-gcs-filesystem>=0.23.1 ; python_version < "3.12"` — so on **Python 3.12 it is simply never installed**, and `tf.saved_model.save(..., "gs://...")` in Step 7.2 dies with `UnimplementedError: File system scheme 'gs' not implemented`. This is not "if it fails"; on a Mac running 3.12 it is a certainty. Pinning it explicitly (there are macOS-arm64 wheels for cp39–cp312) makes the problem disappear before it happens. Note that the *item embeddings* go through `google-cloud-storage` instead, deliberately — that file is the one Vector Search depends on, so it doesn't get to depend on a TensorFlow plugin.

> **⚠️ TFRS does not work with Keras 3** (the default in TF ≥ 2.16). You get opaque failures from *inside* `tfrs.Model` — errors mentioning `keras.src`, or `Unrecognized keras object`. The fix, used everywhere in this guide: install **`tf-keras`** and set **`TF_USE_LEGACY_KERAS=1` before TensorFlow is imported**. The script below sets it defensively on line 2; set it in your shell and in the Dockerfile as well. If you ever see a `keras.src` error, this variable didn't take — it was set *after* the import, which is too late.
>
> To confirm the whole stack landed correctly before you write any code:
> ```bash
> TF_USE_LEGACY_KERAS=1 python -c "import tensorflow as tf, tensorflow_recommenders as tfrs; print(tf.__version__, tf.keras.__name__)"
> # 2.17.0 tf_keras.api._v2.keras     <-- 'tf_keras', not 'keras'. If it says 'keras', the env var didn't take.
> ```

> **`protobuf==4.25.5` is pinned on purpose.** TF 2.17 and the `google-cloud-*` libraries still call `MessageFactory.GetPrototype`, which protobuf 5 removed. Unpinned, you get a stream of `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` — non-fatal here, but noisy enough to mask a *real* BigQuery error underneath it.

> **⚠️ `numpy==1.26.4` and `scipy==1.13.1` are pinned for the same reason, and they are the two that will actually cost you an evening.** **TensorFlow 2.17's wheels are compiled against NumPy 1.x.** Nothing in this file's TF/BigQuery layer will pull NumPy 2 on its own — but `pip install google-cloud-aiplatform` (Step 7.4, and again in Step 15.3's image) **will**, along with a SciPy built for NumPy 2. Neither pin exists to make *this* step work. They exist so that a later, unrelated `pip install` cannot silently break this one.
>
> When it happens, it happens in two acts, and fixing act one just reveals act two:
>
> ```
> # act 1 — you upgraded NumPy
> A module that was compiled using NumPy 1.x cannot be run in NumPy 2.5.1 as it may crash.
> ...
> SystemError: initialization of _pywrap_checkpoint_reader raised unreported exception
>
> # act 2 — you fixed NumPy, but SciPy is still the NumPy-2 build
> UserWarning: A NumPy version >=2.0.0 and <2.8.0 is required for this version of SciPy
> AttributeError: module 'numpy' has no attribute 'long'. Did you mean: 'log'?
> ```
>
> `np.long` was **removed** in NumPy 1.24 and **re-introduced with a different meaning** in NumPy 2. A SciPy compiled for NumPy 2 references it at import time; on NumPy 1.26 that import dies — and because TensorFlow imports SciPy internally (via `keras.engine.training_arrays_v1`), *TensorFlow itself* becomes unimportable. The error names SciPy and NumPy and never mentions the thing you actually did, which was install an unrelated Google SDK.
>
> **The repair, if you're reading this after the fact** — all three in one command, because fixing them one at a time just walks you through the acts:
> ```bash
> pip install "numpy==1.26.4" "scipy==1.13.1" "protobuf==4.25.5"
> python -c "import numpy, scipy, google.protobuf; print(numpy.__version__, scipy.__version__, google.protobuf.__version__)"
> # 1.26.4 1.13.1 4.25.5
> python -c "import tensorflow as tf; print(tf.__version__)"   # 2.17.0, no traceback
> pip check                                                    # anything still conflicting?
> ```

## Step 7.2 — The training script

**`training/train.py`**:

```python
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # MUST precede the tensorflow import

import json
import sys
from datetime import datetime, timezone

import tensorflow as tf
import tensorflow_recommenders as tfrs
from google.cloud import bigquery, storage

PROJECT = os.environ["PROJECT"]
BUCKET = "reco-engine-adv-artifacts"     # EU multi-region — canonical home of everything
VECTORS_BUCKET = "reco-engine-adv-vectors"  # europe-west3 — Vector Search's copy, ONLY
MODEL_DIR = os.environ.get("AIP_MODEL_DIR",
                           f"gs://{BUCKET}/models/twotower/dev")
ITEMS_BLOB = "embeddings/items.json"     # written to BOTH buckets — see the upload below
EMB = 32                                 # MUST equal `dimensions` in Phase 8's index

bq = bigquery.Client(project=PROJECT)

# ── Positives: any click / add_to_cart / purchase is an interaction ──────
pairs = bq.query(f"""
    SELECT user_id, item_id
    FROM `{PROJECT}.reco.events`
    WHERE event_type IN ('click', 'add_to_cart', 'purchase')
""").to_dataframe()

if len(pairs) < 5000:
    sys.exit(f"only {len(pairs)} interaction pairs — let the generator and the "
             f"Dataflow job run longer. 5,000 is the hard floor; 20,000+ is what you want, "
             f"which is roughly 90 minutes of the Phase 2 generator at SLEEP=0.1.")
print(f"{len(pairs)} positive interaction pairs")

# .astype(str) is defensive: BigQuery's Arrow path can hand back nullable/extension
# string dtypes that tf.data refuses to convert into a tf.string tensor.
users = pairs["user_id"].astype(str)
items = pairs["item_id"].astype(str)
user_ids = sorted(users.unique().tolist())
item_ids = sorted(items.unique().tolist())
# NOTE: item_ids comes from INTERACTIONS, not from the catalog. An item nobody has ever
# clicked gets no embedding, so it never enters the Vector Search index and can never be
# retrieved. That is a real cold-start property of two-tower retrieval, not a bug — but it
# means len(item_ids) can be < 500, and everything downstream has to allow for that.
print(f"{len(user_ids)} users, {len(item_ids)} items in the training set "
      f"(catalog is 500 — the difference is items with zero positive interactions)")

ds = (tf.data.Dataset.from_tensor_slices({
        "user_id": users.to_numpy(),
        "item_id": items.to_numpy()})
      .shuffle(200_000, seed=42)
      .batch(512)
      .prefetch(tf.data.AUTOTUNE))
items_ds = tf.data.Dataset.from_tensor_slices(item_ids)


def tower(vocab):
    # StringLookup reserves index 0 for out-of-vocabulary, so the embedding
    # table needs len(vocab) + 1 rows.
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None),
        tf.keras.layers.Embedding(len(vocab) + 1, EMB),
    ])


user_model, item_model = tower(user_ids), tower(item_ids)


class TwoTower(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items_ds.batch(256).map(item_model)))

    def compute_loss(self, feats, training=False):
        return self.task(self.user_model(feats["user_id"]),
                         self.item_model(feats["item_id"]))


model = TwoTower()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(ds, epochs=5)     # watch factorized_top_k/top_100_categorical_accuracy climb

# ── Export the towers with an explicit string-input signature ────────────
# Nothing in this build SERVES these SavedModels (the API uses precomputed
# vectors). They're exported so that "deploy a real-time user tower" is a
# one-liner later — and because an untested export is a lie in your README.
class Exporter(tf.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def __call__(self, ids):
        return self.m(ids)


tf.saved_model.save(Exporter(user_model), f"{MODEL_DIR}/user_tower")
tf.saved_model.save(Exporter(item_model), f"{MODEL_DIR}/item_tower")
print("saved towers →", MODEL_DIR)

# ── Item embeddings → GCS JSONL (the Vector Search ingestion format) ─────
# Written to BOTH buckets, every run:
#   artifacts (EU)  — canonical. refresh_index.py (Phase 15) reads this one.
#   vectors (west3) — Vector Search's contents_delta_uri. It CANNOT read the EU bucket.
# Writing both here, rather than copying by hand before Phase 8, is what stops the
# two from drifting — and a drifted mirror is a silent bug: rebuild the index from a
# stale mirror and you serve last month's embeddings with no error anywhere.
vecs = item_model(tf.constant(item_ids)).numpy()
payload = "\n".join(
    json.dumps({"id": iid, "embedding": [float(x) for x in v]})
    for iid, v in zip(item_ids, vecs)
) + "\n"
gcs = storage.Client(project=PROJECT)
for b in (BUCKET, VECTORS_BUCKET):
    gcs.bucket(b).blob(ITEMS_BLOB).upload_from_string(payload)
    print(f"wrote {len(item_ids)} item embeddings → gs://{b}/{ITEMS_BLOB}")

# ── User embeddings → BigQuery (APPEND with trained_at; the API reads the
#    _latest view. Never TRUNCATE+insert: the streaming buffer blocks DML
#    on freshly-streamed rows for ~90 minutes.) ────────────────────────────
uvecs = user_model(tf.constant(user_ids)).numpy()
now = datetime.now(timezone.utc).isoformat()
rows = [{"user_id": u, "embedding": [float(x) for x in v], "trained_at": now}
        for u, v in zip(user_ids, uvecs)]
for i in range(0, len(rows), 500):
    errs = bq.insert_rows_json(f"{PROJECT}.reco.user_embeddings", rows[i:i + 500])
    if errs:
        raise RuntimeError(errs)
print(f"wrote {len(rows)} user embeddings → reco.user_embeddings")
```

## Step 7.3 — Train locally (free, and fast at this scale)

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate
TF_USE_LEGACY_KERAS=1 PROJECT=reco-engine-adv python training/train.py
```

Minutes, not hours. `factorized_top_k/top_100_categorical_accuracy` climbing across the five epochs is your first evidence the model is learning something.

> **Be honest about that metric.** `factorized_top_k` here is computed on the **training** data — it is not an offline evaluation, and quoting it as one in your README is the kind of thing that gets caught. The real offline metric for this project is **Phase 9's preferred-category share**, measured against ground truth the model never saw. Use that as your headline.

> **Gotcha — `UnimplementedError: File system scheme 'gs' not implemented`.** If you see this, `tensorflow-io-gcs-filesystem` is missing from your venv — which means you skipped the pin in Step 7.1. Fix it in place: `pip install tensorflow-io-gcs-filesystem==0.37.1`, then re-run.

## Step 7.4 — (Optional, and worth doing) Run it as a Vertex AI custom training job

Same `train.py`, but Google's hardware runs it in a container. Three moving parts: **build an image → push it to Artifact Registry → submit a `CustomContainerTrainingJob`**.

> **⚠️ Apple-Silicon arch gotcha — the one that will bite you.** Your Mac is arm64; Vertex AI training runs on x86 (amd64). A plain local `docker build` produces an arm64 image and the job dies with `exec format error`. The clean fix is to **not build locally at all** — let **Cloud Build** build it on Google's x86 machines, so the mismatch never happens. That's the path below.

**`training/Dockerfile`** (build context is `training/`, so the `COPY` lines carry no path prefix):

```dockerfile
FROM python:3.12-slim
ENV TF_USE_LEGACY_KERAS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Same two-step install as your laptop: TFRS with --no-deps, so it can't drag a second
# TensorFlow in. Keeping the container and the venv byte-identical is the whole point.
RUN pip install --no-cache-dir --no-deps tensorflow-recommenders==0.7.3
COPY train.py .
ENTRYPOINT ["python", "train.py"]
```

Create the Artifact Registry repo (one-time — the API is already on from Phase 1):

```bash
gcloud artifacts repositories create reco-training \
  --repository-format=docker \
  --location=europe-west3 \
  --project=reco-engine-adv
```

Build and push with Cloud Build, from inside `training/`:

```bash
cd ~/dev/reco-engine-adv/training
IMAGE=europe-west3-docker.pkg.dev/reco-engine-adv/reco-training/twotower:v1
gcloud builds submit --tag $IMAGE --project reco-engine-adv
```

> **Gotcha — Cloud Build push permission.** If the build fails at the *push* step with `denied: permission`, grant the writer role to the Compute Engine default SA (which Cloud Build runs as) and retry:
> ```bash
> PN=$(gcloud projects describe reco-engine-adv --format="value(projectNumber)")
> gcloud projects add-iam-policy-binding reco-engine-adv \
>   --member="serviceAccount:${PN}-compute@developer.gserviceaccount.com" \
>   --role="roles/artifactregistry.writer"
> ```

**`training/submit_job.py`**:

```python
from google.cloud import aiplatform

PROJECT, REGION = "reco-engine-adv", "europe-west3"
BUCKET = "gs://reco-engine-adv-artifacts"
IMAGE = "europe-west3-docker.pkg.dev/reco-engine-adv/reco-training/twotower:v1"

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

job = aiplatform.CustomContainerTrainingJob(
    display_name="twotower-train",
    container_uri=IMAGE,
)

job.run(
    replica_count=1,
    machine_type="n1-standard-4",       # tiny model — no GPU needed
    environment_variables={
        "PROJECT": PROJECT,             # train.py reads this — without it the job crashes on KeyError
        "AIP_MODEL_DIR": f"{BUCKET}/models/twotower/v1",
        "TF_USE_LEGACY_KERAS": "1",
    },
    service_account="trainer@reco-engine-adv.iam.gserviceaccount.com",
    sync=True,                          # block until the job finishes
)
print("training job complete")
```

Before you submit, grant the trainer service account the BigQuery Storage Read role — **`train.py` cannot read its own training data without it**, and this is the single most likely reason your first job fails. If you built `infra/main.tf` from the current Phase 1, it's already there and this is a no-op; if you built from an earlier copy, add it to `trainer_roles`, `terraform apply`, and confirm:

```bash
gcloud projects get-iam-policy reco-engine-adv \
  --flatten="bindings[].members" \
  --filter="bindings.members:trainer@reco-engine-adv.iam.gserviceaccount.com" \
  --format="value(bindings.role)"
# must include roles/bigquery.readSessionUser AND roles/logging.logWriter
```

Now install the submission SDK and submit:

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate
pip install google-cloud-aiplatform
pip install "protobuf==4.25.5" "numpy==1.26.4" "scipy==1.13.1"   # NOT optional — read the box below
python -c "import numpy, scipy, google.protobuf, tensorflow as tf; \
print(numpy.__version__, scipy.__version__, google.protobuf.__version__, tf.__version__)"
# 1.26.4 1.13.1 4.25.5 2.17.0   <-- any other numbers here and TensorFlow in this venv is broken
python training/submit_job.py
```

> **⚠️ `pip install google-cloud-aiplatform` will break your local TensorFlow, quietly — and in three places, not one.** The SDK depends on `google-genai`, which requires a modern `protobuf` — so pip **uninstalls the `protobuf==4.25.5` you pinned in Step 7.1** and drops protobuf 6.x in its place. In the same pass the resolver also lifts **NumPy to 2.x** and pulls a **SciPy built against NumPy 2**, both of which TensorFlow 2.17 cannot live with (see the pins box in Step 7.1). That is why the re-pin line above names all three. It tells you what it's doing, but only as a wall of resolver text you will scroll past:
> ```
> tensorflow 2.17.0 requires protobuf ... <5.0.0dev, but you have protobuf 6.33.6 which is incompatible.
> ```
> The Vertex job still runs — the *container* has its own dependency set, built from `requirements.txt`, and doesn't care what your laptop looks like. **The damage is therefore invisible until much later**: your **local** `train.py` and the Phase 15 retrain will now die on descriptor/`MessageFactory` errors (protobuf) or on `compiled using NumPy 1.x` / `numpy has no attribute 'long'` (NumPy + SciPy) — days after the install that caused them, in scripts that worked yesterday and that you did not touch. Re-pinning all three afterwards fixes it; the aiplatform SDK submits jobs perfectly well on protobuf 4 and NumPy 1, so ignore the resolver warnings that reappear. This is the price of one venv doing two jobs — and it is exactly why Beam got its own, and why KFP gets a third in Step 15.3.

> **Gotcha — pin `AIP_MODEL_DIR` yourself.** Vertex will otherwise inject its own (`gs://…/aiplatform-custom-training-…/model/`), sending the towers somewhere nothing looks for them. Setting it explicitly above forces a predictable path. The `items.json` paths are *not* affected — they're hardcoded to `embeddings/items.json` in **both** buckets (`…-artifacts` for everything downstream, `…-vectors` for Phase 8's index), and they're written identically whether you trained locally or on Vertex.

> **What "stuck" looks like (it isn't).** The job goes `PIPELINE_STATE_PENDING` (Vertex provisions a machine and pulls your TensorFlow image — several minutes) → `PIPELINE_STATE_RUNNING`. Because `sync=True` only prints on **state changes**, the terminal goes quiet once it's running. That's expected. Submit-to-done is typically 8–15 minutes, mostly provisioning. Only worry if the state flips to `PIPELINE_STATE_FAILED`. Your Mac isn't running anything — the job lives on Vertex, so closing the terminal doesn't kill it, it just costs you the `sync=True` prints. The artifact checks in Checkpoint 7 are the real source of truth.

> **Reading the job's logs — and the `--limit` trap that hides your traceback.** `train.py`'s stdout goes to **Cloud Logging**, not your terminal. But Vertex emits a *"provisioning job running framework"* line **every 5 seconds while it waits for a machine** — that's ~80 identical lines before your container even starts. A naive `--limit=80 --order=asc` therefore returns nothing but provisioning spam and looks like an empty log. Filter it out, and add `jsonPayload.message` to the format (container stdout often lands there rather than in `textPayload`, so a `textPayload`-only format prints blank rows):
>
> ```bash
> JOB=<the numeric id from the "View backing custom job" URL>
> gcloud logging read \
>   "resource.type=\"ml_job\"
>    AND resource.labels.job_id=\"$JOB\"
>    AND NOT textPayload:\"provisioning\"
>    AND NOT textPayload:\"Waiting for job\"" \
>   --project=reco-engine-adv --limit=100 --order=asc \
>   --format="value(timestamp, severity, textPayload, jsonPayload.message)"
> ```
>
> To go straight to a failure, swap in `--order=desc --limit=40` — a Python traceback is always at the *end*, so read it bottom-up. Note that Vertex **retries the replica once** before giving up, so a failed job shows the same traceback **twice**; that's one bug, not two.
>
> If both come back genuinely empty, the container never managed to write logs at all — grant `roles/logging.logWriter` to `trainer@` and re-submit. (You shouldn't need it: Vertex's agent writes custom-job logs on your behalf. But a silently log-less container is a five-minute IAM check versus an hour of guessing.)

> **"Training did not produce a Managed Model" is not an error.** On success the SDK prints:
> ```
> CustomContainerTrainingJob run completed. Resource name: ...
> Training did not produce a Managed Model returning None. Training Pipeline ... is not
> configured to upload a Model. Create the Training Pipeline with model_serving_container_image_uri
> and model_display_name passed in. ...
> training job complete
> ```
> All it means is that `job.run()` returned `None` instead of a `Model` object — because `submit_job.py` deliberately passes **no** `model_serving_container_image_uri` and **no** `model_display_name`. Nothing in this build serves a model from the Vertex **Model Registry**: `train.py` writes its own three artifacts (towers → GCS, item vectors → `items.json`, user vectors → BigQuery), and Vector Search + BQML do the serving. Registering a Managed Model would create a resource nothing reads. The SDK can't know that, so it warns. **`training job complete` on the last line is your success signal** — that, and the artifact checks below.

> **Rebuild after edits.** The image is a frozen snapshot of `train.py`. Change the script and you must re-run `gcloud builds submit` with a bumped tag (`:v2`) *and* update `IMAGE` in `submit_job.py` — otherwise Vertex cheerfully keeps running the old code.

**Cost:** the training job is one-shot and terminates itself. No endpoint, no idle machine, nothing to tear down.

✅ **Checkpoint 7:** training metrics improved across epochs; both towers are in GCS; `embeddings/items.json` exists **in both buckets**; `reco.user_embeddings_latest` returns one 32-float vector per user. **Core Path complete.**

Verify it rather than trusting the terminal — especially if you trained on Vertex, where the success message is buried in a warning:

```bash
# 1. Both towers exported (saved_model.pb + variables/ under each)
gcloud storage ls -r gs://reco-engine-adv-artifacts/models/twotower/v1/

# 2. items.json exists, is NOT zero bytes, and is in BOTH buckets with the SAME size.
#    The second one is what Phase 8's index will actually read.
gcloud storage ls -l gs://reco-engine-adv-artifacts/embeddings/items.json
gcloud storage ls -l gs://reco-engine-adv-vectors/embeddings/items.json

# 3. One JSON object per line, and the vector is exactly 32 long (= EMB = Phase 8's `dimensions`)
gcloud storage cat gs://reco-engine-adv-artifacts/embeddings/items.json | head -1

# 4. User vectors landed in BigQuery, same dimensionality
bq query --use_legacy_sql=false --project_id=reco-engine-adv \
  'SELECT COUNT(*) AS users, ARRAY_LENGTH(ANY_VALUE(embedding)) AS dims
   FROM `reco-engine-adv.reco.user_embeddings_latest`'
```

If you trained on Vertex, also confirm the model actually *learned* — a job can succeed while producing an index full of noise:

```bash
gcloud logging read \
  'resource.type="ml_job" AND resource.labels.job_id="<JOB_ID>" AND textPayload:"factorized_top_k"' \
  --project=reco-engine-adv --limit=20 --order=asc --format="value(textPayload)"
```

`factorized_top_k/top_100_categorical_accuracy` should **climb across the five epochs**. Flat means the model didn't learn, and Phase 8 will happily build a Vector Search index out of nothing. (Same honesty caveat as Step 7.3: this is a *training-set* metric — evidence of learning, not an offline evaluation. Phase 9's preferred-category share is the number for your README.)

---

# PRODUCTION SYSTEM

# Phase 8 — Build & deploy the Vector Search index (~2 hrs, mostly waiting)

**Goal:** Load the item embeddings into **Vertex AI Vector Search** for millisecond nearest-neighbour retrieval.

> **Term — ANN (approximate nearest neighbour):** finding the closest vectors to a query vector *exactly* means comparing against all of them — fine for 500 items, hopeless for 50 million. ANN indexes (Vector Search uses a tree + quantization scheme called ScaNN/tree-AH) trade a sliver of accuracy for a logarithmic-ish search, turning "scan everything" into "look in the right neighbourhood." This is the same machinery behind RAG vector databases — which is why this phase transfers directly to GenAI retrieval work, not just recsys.

**`featurestore/create_index.py`**:

```python
from google.cloud import aiplatform

PROJECT, REGION = "reco-engine-adv", "europe-west3"
aiplatform.init(project=PROJECT, location=REGION)

# STREAM_UPDATE lets Phase 15 upsert refreshed embeddings into the LIVE index
# without a rebuild. Choose BATCH_UPDATE here and refresh_index.py will fail.
# (Class names may read VectorSearch* in newer SDKs — the Matching Engine →
# Vector Search rename. Both spellings appear in current samples.)
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="item-index",
    # The REGIONAL bucket (europe-west3), not the EU multi-region one. See below.
    contents_delta_uri="gs://reco-engine-adv-vectors/embeddings",
    dimensions=32,                                  # MUST equal EMB in train.py
    approximate_neighbors_count=50,
    distance_measure_type="DOT_PRODUCT_DISTANCE",   # matches how TFRS trains its towers
    index_update_method="STREAM_UPDATE",
    # Both of these look optional. They are not — omit both and the SDK sends no
    # algorithmConfig at all and the API rejects the call. See the box below.
    leaf_node_embedding_count=500,                  # 500 items → one leaf
    leaf_nodes_to_search_percent=100,               # scan it all → exact, at this scale
)
print("index:", index.resource_name)

# public_endpoint_enabled=True is ESSENTIAL for this setup. Without it the
# endpoint is reachable only over VPC peering, and find_neighbors() from Cloud
# Run or your laptop simply never connects — it times out with no useful error.
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="item-index-endpoint",
    public_endpoint_enabled=True,
)
print("endpoint:", endpoint.resource_name)

endpoint.deploy_index(index=index, deployed_index_id="items_v1")
print("deployed as items_v1")
```

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate
python featurestore/create_index.py
```

This call fails in **two** distinct ways on a clean project, back to back, at the same line. Both are fixed in the code above; read them anyway, because the second one is a real architectural constraint and the first one is a good lesson in reading a library's source.

> **⚠️ Failure 1 — `algorithmConfig is required but missing from the metadata`.** Omit `leaf_node_embedding_count` **and** `leaf_nodes_to_search_percent` — as any reasonable person would, since both are documented "Optional" with server-side defaults — and the call dies with `FAILED_PRECONDITION: algorithmConfig is required but missing`. Here is why, from `matching_engine_index.py` in `google-cloud-aiplatform` ≥ 1.125:
>
> ```python
> algorithm_config = None
> if (leaf_node_embedding_count is not None
>         or leaf_nodes_to_search_percent is not None):
>     algorithm_config = matching_engine_index_config.TreeAhConfig(...)
> ```
>
> and then, in the serializer: `res["algorithmConfig"] = None` when there's no config — a null that gets stripped when the metadata dict becomes a protobuf `Struct`. So `create_tree_ah_index` sends a request containing **no tree-AH config**, and the server refuses it. Versions ≤ **1.122** built the `TreeAhConfig` unconditionally, passing `None`s that the server filled in itself; the regression landed between 1.122 and 1.125. **Passing either parameter is enough to satisfy it.** Pass both, so the search behaviour is stated rather than inherited.
>
> **Why 500 / 100 specifically.** `leaf_node_embedding_count` is how many vectors sit in each leaf of the tree; at 500 items that's a single leaf. `leaf_nodes_to_search_percent` is what fraction of leaves a query scans — the default 10 % is a speed/recall trade-off that only starts paying for itself at millions of vectors. At 500 items, scanning 100 % costs microseconds and gives you **exact** nearest neighbours, so Phase 9's number measures your *model* and not ANN approximation error. Don't pin the SDK backwards to dodge this (`pip install 'google-cloud-aiplatform<1.125'`) — you'd drag Feature Store, KFP, and prediction back months to fix one line.

> **⚠️ Failure 2 — `The Cloud Storage bucket ... is in location 'eu'. It must be in the same regional location as the service location 'europe-west3'.`** This is why `contents_delta_uri` points at `reco-engine-adv-vectors` and not at the artifacts bucket everything else uses. **`EU` is a multi-region; `europe-west3` is a region. To Vector Search these are different location *types*, not "near enough."** Every other service in this build is happy with the multi-region bucket — BigQuery (also `EU`), Dataflow staging, the tower exports. Vector Search is the sole exception, and it is a hard stop.
>
> The tempting fix — move the artifacts bucket to `europe-west3` — is the wrong one: that bucket is Dataflow's staging location, and Step 3.3's escape ladder (*"Frankfurt is out of machines, relaunch in `europe-west1`"*) works **only** while it spans all of Europe. Pin it to Frankfurt and you trade this failure for a worse, rarer one. Hence Phase 1's second bucket, and hence `train.py` writing `items.json` to both.

> **⚠️ `contents_delta_uri` is a *folder*, and the folder must contain nothing but embedding files.** Index creation reads **every** object under `gs://reco-engine-adv-vectors/embeddings/`. Drop a stray `notes.txt` or a `.DS_Store` in there and the build fails with a parse error on a file you forgot about. Keep that prefix sacred — `items.json` and nothing else. (Anything else you want to keep — archived runs, exports, scratch files — belongs in `archive/` in the *artifacts* bucket, and never anywhere near this one.)

**Before you start the 20–60 minute wait, confirm the mirror is there and current.** Nothing is more annoying than discovering an hour later that you indexed an empty prefix:

```bash
gcloud storage ls -l gs://reco-engine-adv-vectors/embeddings/
# exactly ONE object: items.json, non-zero, and the same size as the artifacts copy
gcloud storage buckets describe gs://reco-engine-adv-vectors --format='value(location)'
# EUROPE-WEST3  ← if this says EU, stop; the create call cannot succeed
```

> **If you trained *before* adding the dual-write to `train.py`** (i.e. you're patching a v4 build), the mirror doesn't exist yet. Either re-run `train.py`, or copy it once by hand:
> ```bash
> gcloud storage cp gs://reco-engine-adv-artifacts/embeddings/items.json \
>                   gs://reco-engine-adv-vectors/embeddings/items.json
> ```
> Then go add the dual-write anyway — a mirror you have to remember to refresh is a mirror that will one day be stale, and a stale mirror produces a perfectly healthy-looking index full of last month's vectors.

> **Patience + cost.** Index creation and the first deploy each take **20–60 minutes**. That is normal, not a hang. Once deployed the endpoint holds a node and **bills while it's up** (single-digit €/day) — this is off-switch #2 from Phase 0.

You do **not** need to copy those resource names by hand — every later phase re-derives them:

```bash
INDEX=$(gcloud ai indexes list --region=europe-west3 \
  --filter='displayName=item-index' --format='value(name)')
INDEX_ENDPOINT=$(gcloud ai index-endpoints list --region=europe-west3 \
  --filter='displayName=item-index-endpoint' --format='value(name)')
echo "$INDEX"
echo "$INDEX_ENDPOINT"    # projects/<NUMBER>/locations/europe-west3/indexEndpoints/<ID>
```

> **Dimension mismatch is unfixable in place.** `dimensions=32` must equal `EMB` in `train.py`. You cannot change the dimensionality of an existing index — if you ever bump `EMB`, you delete the index and rebuild it. Same for `distance_measure_type`: TFRS retrieval scores with a dot product, so `DOT_PRODUCT_DISTANCE` is the matching measure. Use cosine or L2 here and retrieval quietly degrades — no error, just worse recommendations.
>
> **And when you do rebuild: `contents_delta_uri` is read at *creation* and never again.** Because you chose `STREAM_UPDATE`, Phase 15's refreshes go through `upsert_datapoints()` — an API call that never touches GCS. So the `…-vectors` mirror is a **bootstrap fixture**, consulted exactly once per index. That's fine while `train.py` keeps it current. It stops being fine the day you rebuild an index without having retrained: you will index whatever `items.json` was last written, get no error, and serve stale embeddings. **Retrain, then rebuild — in that order.**

✅ **Checkpoint 8:** **Vertex AI → Vector Search** shows `item-index` deployed to `item-index-endpoint`, and the two shell variables above print full resource names.

---

# Phase 9 — Prove retrieval works (~1 hr)

**Goal:** An end-to-end retrieval test with a real, honest metric — for sample users, do the retrieved items match their *ground-truth* preferred category (which the model never saw)?

**`featurestore/test_retrieval.py`**:

```python
import sys

from google.cloud import aiplatform, bigquery

PROJECT, REGION = "reco-engine-adv", "europe-west3"
INDEX_ENDPOINT = sys.argv[1]        # full resource name — see the shell snippet below
DEPLOYED_INDEX_ID = "items_v1"
TOP_K = 10

aiplatform.init(project=PROJECT, location=REGION)
bq = bigquery.Client(project=PROJECT)
ep = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT)

catalog = {r["item_id"]: r["category"] for r in bq.query(
    f"SELECT item_id, category FROM `{PROJECT}.reco.catalog`").result()}
truth = {r["user_id"]: r["preferred_category"] for r in bq.query(
    f"SELECT user_id, preferred_category FROM `{PROJECT}.reco.users_truth`").result()}
uvecs = {r["user_id"]: list(r["embedding"]) for r in bq.query(
    f"SELECT user_id, embedding FROM `{PROJECT}.reco.user_embeddings_latest`").result()}

if not uvecs:
    sys.exit("no user embeddings — run training/train.py first (Phase 7)")

sample = [u for u in list(uvecs) if u in truth][:50]
hits = total = 0
for n, uid in enumerate(sample):
    res = ep.find_neighbors(deployed_index_id=DEPLOYED_INDEX_ID,
                            queries=[uvecs[uid]], num_neighbors=TOP_K)
    cats = [catalog.get(m.id, "?") for m in res[0]]
    hits += sum(c == truth[uid] for c in cats)
    total += TOP_K
    if n < 5:
        print(f"{uid} prefers {truth[uid]:<12} top-{TOP_K} categories: {cats}")

print(f"\npreferred-category share in top-{TOP_K}: {hits / total:.1%} "
      f"(random baseline = 20% with 5 categories)")
```

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate
INDEX_ENDPOINT=$(gcloud ai index-endpoints list --region=europe-west3 \
  --filter='displayName=item-index-endpoint' --format='value(name)')
python featurestore/test_retrieval.py "$INDEX_ENDPOINT"
```

With the 70 %-preference generator, expect the preferred-category share to beat the 20 % random baseline by a wide margin. **That number is your headline offline metric** — it's measured against ground truth the model was never shown, which is exactly what `factorized_top_k` is not.

> **If every user returns the same items, or the share sits at ~20 %:** the index is serving stale or empty content. Confirm the embeddings file is populated and that you created the index *after* training wrote it. Check the **`…-vectors`** copy — that is the one the index was actually built from, and if the two buckets ever disagree, this is where the lie lives:
> ```bash
> gcloud storage cat gs://reco-engine-adv-vectors/embeddings/items.json | wc -l
> ```
> Expect a number **at or just below 500** — one line per item that had at least one positive interaction (see the note in Step 7.2). If it prints a number in the low hundreds, your generator hasn't run long enough; if it prints `0`, training never wrote the file. (`gsutil` still works but is on its way out — `gcloud storage` is the current CLI.)

✅ **Checkpoint 9:** retrieval clearly beats the 20 % random baseline on preferred-category share.

---

# Phase 10 — The ranking model (retrieve → rank) (~2 hrs)

**Goal:** Stage two of the classic recsys architecture. Retrieval narrows 500 items → 50 candidates in milliseconds; a **BQML boosted-tree classifier** then re-orders those 50 using richer features that would be far too expensive to apply to the whole catalog.

> **Term — retrieve → rank:** a two-stage funnel. Stage 1 (cheap, approximate, high recall) throws away 90 % of the catalog. Stage 2 (expensive, precise, high precision) carefully orders what's left. Splitting the problem this way is what makes real recommenders both fast *and* accurate — and knowing *why* the split exists is the whole idea.

## Step 10.1 — Build the training set (positives from events, sampled negatives)

```sql
CREATE OR REPLACE TABLE `reco-engine-adv.reco.rank_training` AS
WITH pos AS (
  SELECT DISTINCT user_id, item_id, 1 AS label
  FROM `reco-engine-adv.reco.events`
  WHERE event_type IN ('click', 'add_to_cart', 'purchase')
),
neg AS (
  -- Random user×item pairs that never happened. Tune the RAND() threshold to
  -- land near a 2:1 negative:positive ratio (see the check below).
  SELECT u.user_id, i.item_id, 0 AS label
  FROM (SELECT DISTINCT user_id FROM `reco-engine-adv.reco.events`) u
  CROSS JOIN (SELECT item_id FROM `reco-engine-adv.reco.catalog`) i
  WHERE RAND() < 0.30
    AND NOT EXISTS (SELECT 1 FROM pos p
                    WHERE p.user_id = u.user_id AND p.item_id = i.item_id)
),
all_rows AS (SELECT * FROM pos UNION ALL SELECT * FROM neg)
SELECT
  r.label,
  uf.events_7d, uf.purchases_30d, uf.distinct_items_7d,
  itf.category, itf.price, itf.popularity_7d,
  CAST(uf.top_category = itf.category AS INT64) AS same_category
FROM all_rows r
JOIN `reco-engine-adv.reco.user_features_latest` uf USING (user_id)    -- _latest, NOT the raw table
JOIN `reco-engine-adv.reco.item_features_latest` itf USING (item_id);  -- (see the fan-out trap, Phase 4)

-- Check the class balance BEFORE training:
SELECT label, COUNT(*) AS n FROM `reco-engine-adv.reco.rank_training` GROUP BY label;
```

> **Two things to notice, both of which matter.**
> **(1) The joins hit `_latest`, not the raw feature tables.** Join the append-only tables and every training row is duplicated once per feature snapshot — a silent, ratio-preserving fan-out that inflates your row count, wrecks nothing visibly, and quietly makes your model overconfident. This is the fan-out trap from Phase 4, and this query is where it would first bite.
> **(2) `user_id` and `item_id` are deliberately absent from the SELECT list.** BQML treats every non-label column as a feature. Leave the IDs in and the model memorizes them instead of learning anything generalizable — the classic recsys leak.
>
> If the balance check comes back badly skewed (say 5:1 either way), adjust `RAND() < 0.30` and re-run. `auto_class_weights = TRUE` below compensates for moderate imbalance, but you want the ratio roughly sane going in.

## Step 10.2 — Train + evaluate

```sql
CREATE OR REPLACE MODEL `reco-engine-adv.reco.ranker`
OPTIONS(
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['label'],
  auto_class_weights = TRUE
) AS
SELECT * FROM `reco-engine-adv.reco.rank_training`;

SELECT * FROM ML.EVALUATE(MODEL `reco-engine-adv.reco.ranker`);
```

## Step 10.3 — Confirm the model found the structure

```sql
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL `reco-engine-adv.reco.ranker`)
ORDER BY importance_gain DESC;
```

`same_category` should dominate — the boosted tree discovering the exact same hidden preference structure the two-tower model found, through a completely different lens. That agreement between two independent models is worth a sentence in your README.

> **The one-paragraph framing:** *"Retrieval — two-tower plus Vector Search — narrows 500 candidates to 50 in milliseconds. Ranking — boosted trees on cross features like user-top-category × item-category — orders those 50. Splitting the problem is what lets a real recommender be both fast and accurate."*

✅ **Checkpoint 10:** the ranker trains with a sane ROC AUC (well above 0.5) and `same_category` tops feature importance.

---

# Phase 11 — The online recommender API (~3 hrs)

**Goal:** One Cloud Run service that ties everything together: **hash the user into a variant → (treatment) fetch their vector from BigQuery → retrieve 50 from Vector Search → rank with BQML → return top-10 → log the impression.** Control gets popular items — the baseline the treatment has to beat.

```bash
mkdir -p ~/dev/reco-engine-adv/api
```

## Step 11.1 — The service

**`api/main.py`** (complete):

```python
import os
import json
import uuid
import hashlib
from datetime import datetime, timezone

from flask import Flask, request
from google.cloud import bigquery, aiplatform

app = Flask(__name__)
PROJECT = os.environ["PROJECT"]
REGION = "europe-west3"
INDEX_ENDPOINT = os.environ["INDEX_ENDPOINT"]                 # full resource name
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "items_v1")

aiplatform.init(project=PROJECT, location=REGION)
bq = bigquery.Client(project=PROJECT)

_vs = None


def vs():
    """Lazy so a cold container doesn't pay for the endpoint handshake until
    the first treatment request actually needs it."""
    global _vs
    if _vs is None:
        _vs = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT)
    return _vs


def variant_for(user_id: str) -> str:
    """Deterministic 50/50 split. Hashing the USER (not the request) means a
    user always lands in the same arm — per-user randomization, which is what
    makes the experiment valid. sha256, not hash(): Python's built-in hash is
    salted per process and would reassign users on every container restart."""
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return "treatment" if h % 2 else "control"


def popular_items(k: int = 10):
    q = f"""SELECT item_id FROM `{PROJECT}.reco.item_features_latest`
            ORDER BY popularity_7d DESC LIMIT {k}"""
    return [r["item_id"] for r in bq.query(q).result()]


def user_vector(user_id: str):
    q = f"""SELECT embedding FROM `{PROJECT}.reco.user_embeddings_latest`
            WHERE user_id = @uid"""
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("uid", "STRING", user_id)])
    rows = list(bq.query(q, job_config=cfg).result())
    return list(rows[0]["embedding"]) if rows else None


def retrieve(vec, k: int = 50):
    res = vs().find_neighbors(deployed_index_id=DEPLOYED_INDEX_ID,
                              queries=[vec], num_neighbors=k)
    return [m.id for m in res[0]]


def rank(user_id: str, candidates, k: int = 10):
    q = f"""
    SELECT item_id,
      (SELECT p.prob FROM UNNEST(predicted_label_probs) p WHERE p.label = 1) AS score
    FROM ML.PREDICT(MODEL `{PROJECT}.reco.ranker`, (
      SELECT
        cand AS item_id,
        uf.events_7d, uf.purchases_30d, uf.distinct_items_7d,
        itf.category, itf.price, itf.popularity_7d,
        CAST(uf.top_category = itf.category AS INT64) AS same_category
      FROM UNNEST(@ids) AS cand
      JOIN `{PROJECT}.reco.item_features_latest` itf ON itf.item_id = cand
      CROSS JOIN (SELECT * FROM `{PROJECT}.reco.user_features_latest`
                  WHERE user_id = @uid) uf))
    ORDER BY score DESC
    LIMIT {k}
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("ids", "STRING", candidates),
        bigquery.ScalarQueryParameter("uid", "STRING", user_id)])
    return [r["item_id"] for r in bq.query(q, job_config=cfg).result()]


def log_impression(uid, variant, served_by, items):
    imp_id = str(uuid.uuid4())
    errs = bq.insert_rows_json(f"{PROJECT}.reco.impressions", [{
        "impression_id": imp_id,
        "user_id": uid,
        "variant": variant,
        "served_by": served_by,
        "items": json.dumps(items),
        "served_at": datetime.now(timezone.utc).isoformat(),
    }])
    if errs:
        print("impression insert errors:", errs)
    return imp_id


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/recommend", methods=["POST"])
def recommend():
    body = request.get_json(silent=True) or {}
    uid = body.get("user_id")
    if not uid:
        return {"error": "user_id required"}, 400

    variant = variant_for(uid)
    served_by = "popular"
    items = []

    if variant == "treatment":
        vec = user_vector(uid)
        if vec is not None:
            try:
                items = rank(uid, retrieve(vec, 50), 10)
                served_by = "personalized"
            except Exception as e:
                # Vector Search undeployed, or the ranker model is missing.
                # Fail LOUD: silently serving popular items to the treatment arm
                # would corrupt the experiment with data you can't identify later.
                print("retrieval/ranking failed:", repr(e))
                return {"error": "recommendation backend unavailable"}, 503

    # Control always lands here. So does a treatment user with no embedding yet
    # (a genuine cold start) and a treatment user whose ranking returned nothing.
    # served_by='popular' TAGS those rows so Phase 13 can measure them instead of
    # pretending they didn't happen.
    if not items:
        items = popular_items()
        served_by = "popular"

    imp_id = log_impression(uid, variant, served_by, items)
    return {"impression_id": imp_id, "variant": variant,
            "served_by": served_by, "items": items}, 200


@app.route("/feedback", methods=["POST"])
def feedback():
    """Append-only feedback events. Deliberately NOT an UPDATE on impressions:
    freshly streamed rows sit in BigQuery's streaming buffer for ~90 minutes and
    DML against them fails with `UPDATE or DELETE statement over table would
    affect rows in the streaming buffer, which is not supported`."""
    body = request.get_json(silent=True) or {}
    if body.get("event") not in ("click", "conversion"):
        return {"error": "event must be click|conversion"}, 400
    errs = bq.insert_rows_json(f"{PROJECT}.reco.feedback", [{
        "impression_id": body.get("impression_id"),
        "user_id": body.get("user_id"),
        "event": body["event"],
        "at": datetime.now(timezone.utc).isoformat(),
    }])
    if errs:
        print("feedback insert errors:", errs)
        return {"error": "insert failed"}, 500
    return {"logged": True}, 200
```

**`api/requirements.txt`**:

```
flask==3.0.3
gunicorn==23.0.0
google-cloud-bigquery==3.25.0
google-cloud-aiplatform==1.71.0
```

> **`gunicorn` is pinned at 23.0.0, not 22.** Gunicorn 22.0.0 carries a request-smuggling CVE (improper handling of conflicting `Transfer-Encoding` headers), fixed in 23. Your API is `--allow-unauthenticated` and on the public internet, so this is not a theoretical concern — and "I pinned past a known CVE in my web server" is a good sentence to have available.

> **On `google-cloud-aiplatform==1.71.0`:** the `MatchingEngineIndex` / `MatchingEngineIndexEndpoint` classes and every method this guide calls on them (`create_tree_ah_index`, `deploy_index`, `find_neighbors`, `upsert_datapoints`) are still present and unchanged on **1.160.0**, the current release. Pinning is about reproducibility, not about avoiding a break — bump it freely, and re-run Phase 9 afterwards to prove nothing moved.

**`api/Dockerfile`**:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app
```

> **The `served_by` column is the honest-experiment fix.** A treatment user with no embedding gets popular items — which is *exactly what the control arm gets*. If you don't record that, those rows dilute your treatment CTR toward the control's and you cannot tell afterwards how many there were. Tagging them lets Phase 13 report the fallback rate openly while still analyzing **intent-to-treat** (everyone assigned to treatment stays in the treatment numbers). Quietly dropping them would be per-protocol analysis, and per-protocol analysis of a non-random dropout is how experiments lie.

## Step 11.2 — Deploy

> **First-`--source`-deploy gotcha (same in every one of these projects):** if the build fails on permissions, grant the Compute Engine default SA `roles/run.builder` once:
> ```bash
> PROJECT_NUMBER=$(gcloud projects describe reco-engine-adv --format='value(projectNumber)')
> gcloud projects add-iam-policy-binding reco-engine-adv \
>   --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
>   --role="roles/run.builder"
> ```

```bash
cd ~/dev/reco-engine-adv/api

INDEX_ENDPOINT=$(gcloud ai index-endpoints list --region=europe-west3 \
  --filter='displayName=item-index-endpoint' --format='value(name)')
echo "$INDEX_ENDPOINT"        # sanity-check: must not be empty

gcloud run deploy reco-api --source . --region europe-west3 \
  --set-env-vars "PROJECT=reco-engine-adv,INDEX_ENDPOINT=${INDEX_ENDPOINT},DEPLOYED_INDEX_ID=items_v1" \
  --service-account reco-api@reco-engine-adv.iam.gserviceaccount.com \
  --allow-unauthenticated
```

The service is public so the Phase 12 simulator can hit it without auth plumbing. **Say that in your README** — "public for demo purposes; in production this sits behind IAP or an API gateway." It is a known trade-off if you name it, and an oversight if someone else has to.

## Step 11.3 — Test both arms

```bash
API_URL=$(gcloud run services describe reco-api --region europe-west3 --format='value(status.url)')
echo "$API_URL"

for U in u0007 u0008 u0011 u0012; do
  curl -s -X POST "$API_URL/recommend" -H "Content-Type: application/json" \
    -d "{\"user_id\":\"$U\"}" | python3 -m json.tool | head -5
done
```

Across four users you should see **both** variants appear. Every treatment response should say `"served_by": "personalized"` — if any say `"popular"`, that user has no embedding (re-run training).

```sql
SELECT variant, served_by, COUNT(*) AS n
FROM `reco-engine-adv.reco.impressions`
GROUP BY variant, served_by;
```

✅ **Checkpoint 11:** both variants return items; every call logs a row in `impressions`; treatment rows are `served_by = 'personalized'`; treatment items skew toward the user's preferred category.

---

# Phase 12 — Close the experiment loop with a simulator (~2 hrs)

**Goal:** Simulated users who click **more when the recommendations actually match their true preference**.

> **The honest design — read this before you write the code.** The simulator encodes **user behaviour**, not variant bias. It never checks which arm it's in. It only asks: *"how many of these 10 items are in the category I actually like?"* and clicks accordingly. Treatment wins only if personalization genuinely surfaces more relevant items than the popularity baseline does. Build the click model the other way — "if treatment, click more" — and you've written a random number generator with a p-value. This distinction is the entire difference between a real experiment and a demo.

**`generator/simulate.py`**:

```python
import os
import json
import random
import time

import requests

API = os.environ["API_URL"].rstrip("/")
ROUNDS = int(os.environ.get("ROUNDS", "3000"))

users = [json.loads(line) for line in open("users.jsonl")]
catalog = {r["item_id"]: r["category"]
           for r in (json.loads(line) for line in open("catalog.jsonl"))}

BASE_CTR = 0.03           # click probability when NOTHING shown is relevant
MATCH_BOOST = 0.30        # + this × (share of the top-10 that matches my preference)
CONV_GIVEN_CLICK = 0.20

errors = 0
for i in range(ROUNDS):
    u = random.choice(users)
    try:
        r = requests.post(f"{API}/recommend",
                          json={"user_id": u["user_id"]}, timeout=30)
        r.raise_for_status()
        rec = r.json()
    except Exception as e:
        errors += 1
        if errors <= 3:
            print("recommend failed:", repr(e))
        if errors > 50:
            raise SystemExit("too many failures — is the index endpoint still deployed?")
        time.sleep(1)
        continue

    items = rec["items"]
    if isinstance(items, str):                  # defensive: accept either shape
        items = json.loads(items)

    match_share = (sum(catalog.get(x) == u["preferred_category"] for x in items)
                   / max(len(items), 1))
    p_click = BASE_CTR + MATCH_BOOST * match_share

    if random.random() < p_click:
        fb = {"impression_id": rec["impression_id"], "user_id": u["user_id"],
              "event": "click"}
        requests.post(f"{API}/feedback", json=fb, timeout=30)
        if random.random() < CONV_GIVEN_CLICK:
            fb["event"] = "conversion"
            requests.post(f"{API}/feedback", json=fb, timeout=30)

    if i % 200 == 0:
        print(f"{i}/{ROUNDS}  (errors so far: {errors})")
    time.sleep(0.02)

print(f"done — {ROUNDS} rounds, {errors} errors")
```

```bash
cd ~/dev/reco-engine-adv/generator && source ../.venv/bin/activate
pip install requests

# Re-derive API_URL — don't assume it's still exported from Phase 11:
API_URL=$(gcloud run services describe reco-api --region europe-west3 --format='value(status.url)')
echo "$API_URL"

API_URL="$API_URL" ROUNDS=3000 python simulate.py     # ~10–15 min
```

> **Prerequisites this script silently depends on:** the **Vector Search index endpoint must still be deployed** (if you undeployed it to save money, the API returns 503 and the simulator will bail out after 50 failures — which is the intended behaviour, not a bug), and `users.jsonl` / `catalog.jsonl` must be in the current directory. Those two files are `.gitignore`d, but `generate.py` is seeded — re-running it regenerates byte-identical files.

> **Why 3,000 rounds.** You need enough impressions per arm for the z-test in Phase 13 to have any power. 3,000 rounds ≈ 1,500 per arm; at CTRs around 9 % vs 24 % that's comfortably enough to reach significance. If Phase 13 comes back "not yet significant," the fix is more rounds — not a different test.

✅ **Checkpoint 12:** thousands of impressions across both variants, with feedback rows accumulating in `reco.feedback`.

---

# Phase 13 — Statistical lift analysis (~1.5 hrs)

**Goal:** Decide whether treatment beats control **with statistics**, and state the result the way a data scientist would — with a confidence interval on the *difference*, not just a bigger-looking number.

> **Term — two-proportion z-test:** you have two conversion rates measured on two samples. The z-test asks: *if the two arms were truly identical, how unlikely would a gap this large be, purely by chance?* That probability is the **p-value**. Small p (< 0.05 by convention) → "chance alone is a poor explanation." It does **not** mean the effect is large or important — only that it's unlikely to be noise.

## Step 13.1 — The topline numbers (impressions ⋈ feedback)

```sql
WITH clicks AS (
  SELECT DISTINCT impression_id FROM `reco-engine-adv.reco.feedback` WHERE event = 'click'
),
convs AS (
  SELECT DISTINCT impression_id FROM `reco-engine-adv.reco.feedback` WHERE event = 'conversion'
)
SELECT
  i.variant,
  i.served_by,
  COUNT(*) AS impressions,
  COUNTIF(c.impression_id IS NOT NULL) AS clicks,
  ROUND(SAFE_DIVIDE(COUNTIF(c.impression_id IS NOT NULL), COUNT(*)), 4) AS ctr,
  COUNTIF(v.impression_id IS NOT NULL) AS conversions,
  ROUND(SAFE_DIVIDE(COUNTIF(v.impression_id IS NOT NULL), COUNT(*)), 4) AS cvr
FROM `reco-engine-adv.reco.impressions` i
LEFT JOIN clicks c USING (impression_id)
LEFT JOIN convs  v USING (impression_id)
GROUP BY variant, served_by
ORDER BY variant, served_by;
```

`DISTINCT impression_id` in those CTEs is not decoration: without it, an impression that somehow logged two clicks would be counted twice and your CTR could exceed 100 %. Deduplicate at the join, always.

Expect three rows: `control/popular`, `treatment/personalized`, and — if any user was cold — a small `treatment/popular`. That third row is the fallback rate you're about to report honestly.

## Step 13.2 — The significance test

```bash
mkdir -p ~/dev/reco-engine-adv/analysis
cd ~/dev/reco-engine-adv && source .venv/bin/activate
pip install statsmodels
```

**`analysis/ab_test.py`**:

```python
import sys

from google.cloud import bigquery
from statsmodels.stats.proportion import (
    proportions_ztest, proportion_confint, confint_proportions_2indep)

PROJECT = "reco-engine-adv"
bq = bigquery.Client(project=PROJECT)

q = f"""
WITH clicks AS (
  SELECT DISTINCT impression_id
  FROM `{PROJECT}.reco.feedback` WHERE event = 'click'
)
SELECT
  i.variant,
  COUNT(*) AS n,
  COUNTIF(c.impression_id IS NOT NULL) AS clicks,
  COUNTIF(i.served_by = 'popular') AS fallbacks
FROM `{PROJECT}.reco.impressions` i
LEFT JOIN clicks c USING (impression_id)
GROUP BY variant
"""
d = {r["variant"]: dict(r) for r in bq.query(q).result()}

missing = [v for v in ("control", "treatment") if v not in d]
if missing:
    sys.exit(f"no impressions for {missing} — run the Phase 12 simulator longer")

t, c = d["treatment"], d["control"]
tc, tn = t["clicks"], t["n"]
cc, cn = c["clicks"], c["n"]
if min(tn, cn) < 100:
    print(f"WARNING: only {min(tn, cn)} impressions in the smaller arm — "
          f"any result here is noise. Run more rounds.\n")

t_ctr, c_ctr = tc / tn, cc / cn
stat, pval = proportions_ztest([tc, cc], [tn, cn])
t_lo, t_hi = proportion_confint(tc, tn, method="wilson")
c_lo, c_hi = proportion_confint(cc, cn, method="wilson")
d_lo, d_hi = confint_proportions_2indep(tc, tn, cc, cn, compare="diff")

print(f"control    CTR: {c_ctr:.4f}  ({cc}/{cn})   95% CI [{c_lo:.4f}, {c_hi:.4f}]")
print(f"treatment  CTR: {t_ctr:.4f}  ({tc}/{tn})   95% CI [{t_lo:.4f}, {t_hi:.4f}]")
print()
print(f"absolute difference: {t_ctr - c_ctr:+.4f}   95% CI [{d_lo:+.4f}, {d_hi:+.4f}]")
print(f"relative lift:       {t_ctr / c_ctr - 1:+.1%}")
print(f"z = {stat:.3f}   p = {pval:.5f}")
print("SIGNIFICANT at alpha=0.05" if pval < 0.05 else
      "not yet significant — run the simulator for more samples")
print()
print(f"cold-start fallbacks inside treatment: {t['fallbacks']}/{tn} "
      f"({t['fallbacks'] / tn:.1%}) — kept IN the numbers above (intent-to-treat)")
```

```bash
python analysis/ab_test.py
```

## Step 13.3 — Say the caveats before anyone asks

> **The headline claim:** *"Treatment lifted CTR X % over a popular-items control — absolute difference d, 95 % CI [lo, hi], z = …, p = …, significant at 95 %."* Then earn the rest of the points by **volunteering the caveats**:
> - **The CI that matters is on the difference**, not on either arm alone. Two overlapping per-arm CIs do *not* imply a non-significant difference — a genuinely common mistake, which is why `confint_proportions_2indep` is in the script.
> - **Peeking / early stopping.** Running the test repeatedly and stopping the moment p dips below 0.05 inflates your false-positive rate badly. Fix the sample size first, then look.
> - **Randomization unit.** You randomized per **user** (the sha256 hash), not per impression. Per-impression randomization would show the same user both arms and break independence.
> - **Intent-to-treat.** Cold-start fallbacks stay in the treatment numbers. Dropping them because they "aren't really treatment" is per-protocol analysis of a non-random subgroup — and it biases the result in your favour, which is exactly why it's tempting.
> - **Statistical vs practical significance.** With enough traffic, everything is significant. The effect size and its CI are what a business decides on.

✅ **Checkpoint 13:** a lift number with a p-value, a confidence interval **on the difference**, an explicit significant/not-yet verdict, and a stated fallback rate.

---

# Phase 14 — Drift & model-health monitoring (~1.5 hrs)

**Goal:** Recommenders rot — catalogs churn, tastes shift, embeddings go stale. Track it in SQL, daily.

> **Term — drift:** the world moves away from the data your model was trained on. **Feature drift** = the *inputs* change distribution (users get busier, an item goes viral). **Performance drift** = the *outputs* get worse (CTR sags). Feature drift is the leading indicator; performance drift is the bill arriving.

## Step 14.1 — Model health: CTR per variant per day

```sql
CREATE OR REPLACE TABLE `reco-engine-adv.reco.model_health` AS
WITH clicks AS (
  SELECT DISTINCT impression_id
  FROM `reco-engine-adv.reco.feedback` WHERE event = 'click'
)
SELECT
  DATE(i.served_at) AS day,
  i.variant,
  COUNT(*) AS impressions,
  COUNTIF(i.served_by = 'popular') AS fallbacks,
  ROUND(SAFE_DIVIDE(COUNTIF(c.impression_id IS NOT NULL), COUNT(*)), 4) AS ctr
FROM `reco-engine-adv.reco.impressions` i
LEFT JOIN clicks c USING (impression_id)
GROUP BY day, variant;
```

Watch for three things, each of which means something different:
- a sustained **treatment CTR decline** → the embeddings are stale (retrain: Phase 15);
- the **treatment/control gap closing** → personalization is decaying toward the baseline, which is the same diagnosis with a scarier name;
- **rising fallbacks** → new users are arriving faster than training runs (a genuine cold-start problem, not a drift problem).

## Step 14.2 — Feature drift on `user_features`

This is where the append-only design pays off: you have a *history* of feature snapshots, so drift is just a query.

```sql
CREATE OR REPLACE TABLE `reco-engine-adv.reco.feature_drift` AS
WITH bounds AS (
  -- NOTE: this reads the RAW append-only table on purpose — it is one of the
  -- only two places in the guide that wants the full history, not _latest.
  SELECT MIN(feature_timestamp) AS first_ts, MAX(feature_timestamp) AS last_ts
  FROM `reco-engine-adv.reco.user_features`
),
base AS (
  SELECT
    AVG(events_7d)         AS mu_events, STDDEV(events_7d)         AS sd_events,
    AVG(purchases_30d)     AS mu_purch,  STDDEV(purchases_30d)     AS sd_purch,
    AVG(distinct_items_7d) AS mu_items,  STDDEV(distinct_items_7d) AS sd_items
  FROM `reco-engine-adv.reco.user_features`, bounds
  WHERE feature_timestamp = bounds.first_ts
),
cur AS (
  SELECT
    AVG(events_7d)         AS mu_events,
    AVG(purchases_30d)     AS mu_purch,
    AVG(distinct_items_7d) AS mu_items
  FROM `reco-engine-adv.reco.user_features`, bounds
  WHERE feature_timestamp = bounds.last_ts
)
SELECT 'events_7d' AS feature,
       ROUND(base.mu_events, 3) AS baseline_mean,
       ROUND(cur.mu_events, 3)  AS current_mean,
       ROUND(SAFE_DIVIDE(cur.mu_events - base.mu_events, NULLIF(base.sd_events, 0)), 3) AS z_shift
FROM base, cur
UNION ALL
SELECT 'purchases_30d',
       ROUND(base.mu_purch, 3), ROUND(cur.mu_purch, 3),
       ROUND(SAFE_DIVIDE(cur.mu_purch - base.mu_purch, NULLIF(base.sd_purch, 0)), 3)
FROM base, cur
UNION ALL
SELECT 'distinct_items_7d',
       ROUND(base.mu_items, 3), ROUND(cur.mu_items, 3),
       ROUND(SAFE_DIVIDE(cur.mu_items - base.mu_items, NULLIF(base.sd_items, 0)), 3)
FROM base, cur;

SELECT * FROM `reco-engine-adv.reco.feature_drift` ORDER BY ABS(z_shift) DESC;
```

`z_shift` = how many baseline standard deviations the mean has moved. **|z| > 3 is worth investigating.**

> **Two things about your first drift run.**
> **(1) With only one snapshot it returns three rows of nothing.** `MIN(feature_timestamp)` and `MAX(feature_timestamp)` are then the *same* timestamp, so `base` and `cur` are the same numbers and every `z_shift` comes back as exactly `0.0`. It does **not** error and it does **not** return zero rows — it returns a perfectly-formatted table that says "no drift," which is the most dangerous kind of wrong. You need at least **two** runs of the Phase 4 inserts before this query means anything (they're scheduled hourly; or just run Step 4.1 by hand a second time and re-run this). Confirm first:
> ```sql
> SELECT COUNT(DISTINCT feature_timestamp) AS snapshots FROM `reco-engine-adv.reco.user_features`;
> -- must be >= 2
> ```
> **(2) Your first *real* drift signal will be the generator warming up.** `events_7d` climbs steadily while the event log fills, so you'll see a big positive `z_shift` on it. That is not a false positive — it is a genuine distribution shift, and it's exactly what drift feels like in production. It's also why a real system pins its baseline to **the snapshot the model was trained on** rather than "the oldest row we have." Say that in your README; it's the difference between running a drift query and understanding one.

## Step 14.3 — Schedule it

Console → **BigQuery → Scheduled queries** → both queries above, **daily**.

> **Vertex AI Model Monitoring** is the managed equivalent — but note that it monitors *endpoint-served* models, and this system's "model surface" is an ANN index plus a BQML model, neither of which is a Vertex endpoint. These SQL checks are the honest, runnable demonstration of the same instinct, applied to the architecture you actually built.

✅ **Checkpoint 14:** `model_health` trends CTR per variant per day, and `feature_drift` returns a `z_shift` per feature.

---

# Phase 15 — Retrain + refresh the live index (~2 hrs)

**Goal:** Close the lifecycle. Retrain the two-tower on fresh events, then **upsert** the new item embeddings into the *live* Vector Search index — no rebuild, no redeploy, no downtime. (This works only because Phase 8 chose `STREAM_UPDATE`.)

## Step 15.1 — The refresher

**`featurestore/refresh_index.py`**:

```python
import json
import sys

from google.cloud import aiplatform, storage

PROJECT, REGION = "reco-engine-adv", "europe-west3"
# The EU (artifacts) bucket, deliberately — NOT the europe-west3 vectors mirror.
# This script downloads the vectors itself and pushes them through the API with
# upsert_datapoints(); no GCS location constraint applies. The regional mirror only
# exists for contents_delta_uri, which is read at index CREATION and never again.
BUCKET, BLOB = "reco-engine-adv-artifacts", "embeddings/items.json"
INDEX = sys.argv[1]      # full index resource name

aiplatform.init(project=PROJECT, location=REGION)
index = aiplatform.MatchingEngineIndex(index_name=INDEX)

blob = storage.Client(project=PROJECT).bucket(BUCKET).blob(BLOB)
dps = []
for line in blob.download_as_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    dps.append({"datapoint_id": r["id"], "feature_vector": r["embedding"]})

if not dps:
    sys.exit(f"gs://{BUCKET}/{BLOB} is empty — run training/train.py first")

for i in range(0, len(dps), 100):
    index.upsert_datapoints(datapoints=dps[i:i + 100])
    print(f"  upserted {min(i + 100, len(dps))}/{len(dps)}")
print(f"done — {len(dps)} datapoints refreshed in the live index")
```

> **`upsert_datapoints` only works on a `STREAM_UPDATE` index.** On a `BATCH_UPDATE` index it fails with a `FailedPrecondition`, and your only options are a full rebuild or index aliasing. That one keyword in Phase 8 is what makes this phase a two-minute script instead of an hour-long rebuild — which is a genuinely good "why did you choose that?" answer.

## Step 15.2 — The retraining loop

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate

INDEX=$(gcloud ai indexes list --region=europe-west3 \
  --filter='displayName=item-index' --format='value(name)')
INDEX_ENDPOINT=$(gcloud ai index-endpoints list --region=europe-west3 \
  --filter='displayName=item-index-endpoint' --format='value(name)')

# 1) retrain on fresh events → new item vectors in GCS, new user vectors in BQ
TF_USE_LEGACY_KERAS=1 PROJECT=reco-engine-adv python training/train.py

# 2) push the new item vectors into the LIVE index
python featurestore/refresh_index.py "$INDEX"

# 3) gate: did offline recall survive the retrain?
python featurestore/test_retrieval.py "$INDEX_ENDPOINT"
```

User vectors refresh with no extra step — `train.py` appends to `user_embeddings` with a new `trained_at`, and the API reads `user_embeddings_latest`.

> **⚠️ Step 1 runs `train.py` *locally*, in `.venv`** — not on Vertex. So it needs the Step 7.1 pins intact, and **this is the step where you find out they aren't.** If you did the optional Vertex job in Step 7.4, `pip install google-cloud-aiplatform` rewrote three of them: `protobuf` 4 → 6, `numpy` 1 → 2, and `scipy` → a NumPy-2 build. The retrain then dies on an error that has nothing to do with retraining, and it dies in **two acts** — fix the first and you unlock the second:
>
> ```
> # act 1
> A module that was compiled using NumPy 1.x cannot be run in NumPy 2.5.1 as it may crash.
> SystemError: initialization of _pywrap_checkpoint_reader raised unreported exception
>
> # act 2, after you downgrade NumPy alone
> UserWarning: A NumPy version >=2.0.0 and <2.8.0 is required for this version of SciPy
> AttributeError: module 'numpy' has no attribute 'long'. Did you mean: 'log'?
> ```
>
> Both traces point at NumPy, SciPy and TensorFlow, and none of them mentions the Google SDK that actually caused it. **Repair all three in one command**, then verify before you re-run:
>
> ```bash
> pip install "numpy==1.26.4" "scipy==1.13.1" "protobuf==4.25.5"
> python -c "import numpy, scipy, google.protobuf; print(numpy.__version__, scipy.__version__, google.protobuf.__version__)"
> # 1.26.4 1.13.1 4.25.5
> python -c "import tensorflow as tf; print(tf.__version__)"   # 2.17.0, clean
> pip check                                                    # surfaces anything else the resolver moved
> ```
>
> If `training/requirements.txt` carries the pins (it does, from v6 onward), a plain `pip install -r training/requirements.txt` also repairs it — the pins are re-asserted last, which is the whole reason they're in the file.

> **⚠️ The embedding-space consistency window — the subtle bug hiding in this loop.** Every retrain produces a **brand-new, randomly-initialized embedding space**. A user vector from run 2 is meaningless against item vectors from run 1 — they're coordinates in different rooms. But `train.py` publishes the new **user** vectors to BigQuery *the moment it finishes*, while the new **item** vectors don't reach the index until `refresh_index.py` completes. In that gap — a minute or two — the live API is matching new user vectors against old item vectors, and serving genuinely random recommendations to real traffic.
>
> At portfolio scale: run steps 1 and 2 back to back and don't run the simulator during the gap. In production you'd solve it properly with **index aliasing** (build the new index alongside the old one, swap the alias atomically once both halves are ready) — which is why "blue/green index swaps" is in the "what I'd do next" list, and why being able to explain *this specific failure mode* is worth more than the fix itself.
>
> **And the reason you'd never notice it from your dashboard:** the BQML ranker's strongest feature, `same_category`, is built from the *feature store* (`top_category` × `category`), not from embeddings. So when the embedding spaces disagree, stage 2 keeps working perfectly and quietly rescues roughly a fifth of the garbage stage 1 hands it. Retrieval quality falls toward chance; served quality merely *sags*. Force the two spaces apart deliberately — rebuild the index from a stale `items.json` while fresh user vectors sit in BigQuery — and measure both stages: preferred-category share at retrieval collapses toward the 20 % random baseline, while the served top-10 stays high enough that a dashboard watching *it* would barely flinch. **Monitor stage-1 recall separately, or this bug is invisible to you.**

## Step 15.3 — Wrap the loop in a Vertex AI Pipeline with a recall gate (~2 hrs)

The three commands in 15.2 are a *procedure you remember to run correctly*. This turns them into a **KFP v2 pipeline on Vertex AI** with a `dsl.If` gate in the middle: the new embeddings only reach the live index if they pass an offline recall check. If they don't, the pipeline stops before the upsert and the index keeps serving the old, working vectors.

It's the same conditional-deploy pattern as the predictive-maintenance guide's Phase 7, with a different metric in the gate.

### The design decision that makes this honest — read before writing code

The obvious gate is *"run `test_retrieval.py` after training; promote if the score is good."* **It cannot work, and understanding why is most of the value in this step.**

`test_retrieval.py` queries the **live index** — whose item vectors are still from the *previous* run — using the **new** user vectors from BigQuery. That is precisely the consistency window described in 15.2: two different embedding spaces. It would score ~17 % (chance) after **every** retrain, good or bad, and the gate would never open. A gate wired to that metric isn't strict; it's broken.

So the evaluator must score **off-index**: brute-force the *new* item vectors against the *new* user vectors, in memory. 500 items × 32 dims is a matrix multiply — 16,000 multiply-adds, sub-millisecond. Same metric (preferred-category share in top-10), same ground truth, one self-consistent embedding space. That's `evaluate.py` below. (This is the same "ANN exists to make retrieval *scale*, not to make it *work*" observation the demo layer leans on — here it buys you an evaluation that doesn't depend on the thing you're trying to gate.)

Second decision: **the gate rule is "don't promote a regression," not "only promote an improvement."** Strict improvement sounds rigorous and is a trap — once the model saturates on this data, a perfectly healthy retrain lands within noise of the previous run, and a strict rule means you never promote again. The rule below passes if the score clears an absolute floor **and** hasn't dropped more than a small margin below the reigning champion. Being able to explain *why* the obvious rule is wrong is the point here.

### Step 15.3.1 — A third venv (do not skip this)

`kfp` pulls the same NumPy-2/protobuf-6 family that broke `.venv` in Step 7.4. Give it its own environment, exactly like Beam got one.

```bash
cd ~/dev/reco-engine-adv
/opt/homebrew/bin/python3.12 -m venv .venv-kfp
source .venv-kfp/bin/activate

pip install "kfp==2.17.0" "google-cloud-aiplatform>=1.71.0"
python -c "import kfp; print(kfp.__version__)"    # 2.17.0
deactivate
```

From here on: **`.venv-kfp` compiles and submits. `.venv` is not used in this step at all** — the training runs in a container on Google's machines.

### Step 15.3.2 — The evaluator

**`pipelines/evaluate.py`**:

```python
"""Offline recall gate. Scores the FRESH embedding space (new item vectors from GCS
x new user vectors from BigQuery) by brute force — deliberately never touching the live
index, whose item vectors are still from the previous run. Scoring against the index here
would measure the consistency window, not the model, and would return chance every time."""
import argparse
import json
import os
import sys

import numpy as np
from google.cloud import bigquery, storage

PROJECT = os.environ["PROJECT"]
BUCKET = "reco-engine-adv-artifacts"          # EU multi-region — the canonical copy
ITEMS_BLOB = "embeddings/items.json"
CHAMPION_BLOB = "eval/champion.json"          # the reigning score; written only on promote
TOP_K = 10
N_USERS = 50

FLOOR = float(os.environ.get("RECALL_FLOOR", "0.35"))    # absolute minimum; 0.20 = chance
MARGIN = float(os.environ.get("RECALL_MARGIN", "0.03"))  # tolerated regression vs champion

ap = argparse.ArgumentParser()
ap.add_argument("--decision-out", required=True)
ap.add_argument("--score-out", required=True)
args = ap.parse_args()

bq = bigquery.Client(project=PROJECT)
gcs = storage.Client(project=PROJECT)
bucket = gcs.bucket(BUCKET)

# ── 1. New item vectors, from the file train.py just wrote ───────────────────
rows = [json.loads(l) for l in
        bucket.blob(ITEMS_BLOB).download_as_text().splitlines() if l.strip()]
if not rows:
    sys.exit(f"gs://{BUCKET}/{ITEMS_BLOB} is empty — training produced nothing")

item_ids = [r["id"] for r in rows]
item_mat = np.array([r["embedding"] for r in rows], dtype=np.float32)   # (N, 32)
print(f"{len(item_ids)} item vectors, dim {item_mat.shape[1]}")

# ── 2. New user vectors + ground truth ──────────────────────────────────────
# Row iteration, NOT .to_dataframe() — this container has no need of the Storage
# Read API, so don't reach for a code path that requires a whole extra IAM role.
uvecs = {r["user_id"]: np.array(r["embedding"], dtype=np.float32) for r in bq.query(
    f"SELECT user_id, embedding FROM `{PROJECT}.reco.user_embeddings_latest`").result()}
truth = {r["user_id"]: r["preferred_category"] for r in bq.query(
    f"SELECT user_id, preferred_category FROM `{PROJECT}.reco.users_truth`").result()}
catalog = {r["item_id"]: r["category"] for r in bq.query(
    f"SELECT item_id, category FROM `{PROJECT}.reco.catalog`").result()}

if not uvecs:
    sys.exit("no user embeddings — train.py did not write to reco.user_embeddings")

# ── 3. Brute-force top-K. DOT PRODUCT, to match the index's distance measure ─
sample = sorted(u for u in uvecs if u in truth)[:N_USERS]
hits = total = 0
for uid in sample:
    scores = item_mat @ uvecs[uid]                    # (N,)
    top = np.argsort(-scores)[:TOP_K]                 # highest dot product first
    cats = [catalog.get(item_ids[i], "?") for i in top]
    hits += sum(c == truth[uid] for c in cats)
    total += TOP_K

score = hits / total
print(f"\npreferred-category share in top-{TOP_K}: {score:.1%}  "
      f"(chance = 20%, floor = {FLOOR:.0%})")

# ── 4. Compare against the champion ─────────────────────────────────────────
champ_blob = bucket.blob(CHAMPION_BLOB)
champion = None
if champ_blob.exists():
    champion = float(json.loads(champ_blob.download_as_text())["score"])
    print(f"champion score: {champion:.1%}")
else:
    print("no champion on record — this run sets the baseline")

if score < FLOOR:
    decision, why = "hold", f"{score:.1%} is below the {FLOOR:.0%} floor"
elif champion is not None and score < champion - MARGIN:
    decision, why = "hold", (f"{score:.1%} regressed more than {MARGIN:.0%} "
                             f"below champion {champion:.1%}")
else:
    decision, why = "promote", "above floor, no regression"

print(f"\nDECISION: {decision.upper()} — {why}")

if decision == "promote":
    champ_blob.upload_from_string(json.dumps({"score": score}))
    print(f"champion updated → {score:.1%}")

# ── 5. Hand the results back to KFP ─────────────────────────────────────────
for path, value in ((args.decision_out, decision), (args.score_out, str(score))):
    os.makedirs(os.path.dirname(path), exist_ok=True)   # KFP does not always pre-create it
    with open(path, "w") as f:
        f.write(value)
```

**`pipelines/hold.py`** — the else-branch. It exists so that a rejected retrain shows up **red** in the Vertex UI instead of quietly green.

```python
import sys

score = float(sys.argv[1])
print("=" * 74)
print(f"GATE CLOSED — new embeddings scored {score:.1%} and were NOT promoted.")
print("The live Vector Search index still holds the PREVIOUS item vectors.")
print()
print("⚠️  BUT: train.py already appended new USER vectors to BigQuery, and the API")
print("    reads user_embeddings_latest. The system is now IN the consistency window:")
print("    new user vectors vs. old item vectors. Retrieval quality is at chance.")
print("    Investigate, then either fix and re-run, or upsert manually to close it.")
print("=" * 74)
sys.exit(1)   # non-zero → the pipeline is marked FAILED. That is the point.
```

> **Be honest about this limitation in your README — it is the most interesting thing in the phase.** The gate protects the **index**: the expensive, shared, slow-to-roll-back thing. It does **not** protect the user-embeddings table, because `train.py` appends there unconditionally the moment it finishes. So a *closed* gate leaves you inside the very consistency window the gate exists to reason about. Fixing it properly means staging user vectors to a temp table and swapping the `_latest` view only on promote — which is the same shape of problem as blue/green index aliasing, and belongs in the same "what I'd do next" list. **Shipping a gate while being able to state precisely what it doesn't cover is a stronger signal than shipping one and implying it covers everything.**

### Step 15.3.3 — One image, four scripts

All three stages run in containers on Vertex. Rather than three images, build one that carries everything.

**`pipelines/Dockerfile`** — note the build context will be the **repo root**, so the `COPY` paths carry prefixes:

```dockerfile
FROM python:3.12-slim
ENV TF_USE_LEGACY_KERAS=1
WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps tensorflow-recommenders==0.7.3

# refresh_index.py needs the aiplatform SDK — which will yank protobuf to 6.x and NumPy
# to 2.x and break TensorFlow inside this image, exactly as it did in your venv in Step
# 7.4. Re-assert the pins immediately afterwards. Order matters; do not reorder these.
RUN pip install --no-cache-dir google-cloud-aiplatform
RUN pip install --no-cache-dir "protobuf==4.25.5" "numpy==1.26.4" "scipy==1.13.1"

# Fail the BUILD, not the pipeline at 3am, if the pins didn't survive.
RUN python -c "import numpy, scipy, google.protobuf, tensorflow as tf; \
import tensorflow_recommenders, google.cloud.aiplatform; \
print('deps ok:', numpy.__version__, google.protobuf.__version__, tf.__version__)"

COPY training/train.py .
COPY featurestore/refresh_index.py .
COPY pipelines/evaluate.py .
COPY pipelines/hold.py .
```

No `ENTRYPOINT` — each KFP component supplies its own command.

**`cloudbuild.yaml`** (repo **root**). You need this because `gcloud builds submit --tag` only looks for a `Dockerfile` *at the context root*, and these four scripts live in three different folders:

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - pipelines/Dockerfile
      - -t
      - europe-west3-docker.pkg.dev/reco-engine-adv/reco-training/reco-ops:v1
      - .
images:
  - europe-west3-docker.pkg.dev/reco-engine-adv/reco-training/reco-ops:v1
```

**`.gcloudignore`** (repo **root**) — ⚠️ **not optional.** Without it, `gcloud builds submit` tars up and uploads your venvs — several GB of TensorFlow — to Cloud Build on **every single build**. (Step 7.4 escaped this because its context was `training/`, which contains nothing large.)

```
.venv/
.venv-beam/
.venv-kfp/
__pycache__/
*.pyc
.git/
.terraform/
*.tfstate*
logs/
```

Build it:

```bash
cd ~/dev/reco-engine-adv
gcloud builds submit --config cloudbuild.yaml --project reco-engine-adv .
```

Watch the log for `deps ok: 1.26.4 4.25.5 2.17.0`. If the build dies on that `RUN python -c` line, the pins lost the fight — which is exactly what you asked that line to tell you.

### Step 15.3.4 — The pipeline

**`pipelines/reco_pipeline.py`**:

```python
from kfp import compiler, dsl

PROJECT, REGION = "reco-engine-adv", "europe-west3"
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT}/reco-training/reco-ops:v1"


@dsl.container_component
def train():
    """Retrain the two-tower. Writes towers → GCS, items.json → both buckets,
    user vectors → BigQuery (appended with a fresh trained_at)."""
    return dsl.ContainerSpec(image=IMAGE, command=["python", "/app/train.py"])


@dsl.container_component
def evaluate(decision: dsl.OutputPath(str), score: dsl.OutputPath(float)):
    """Brute-force offline recall on the FRESH embedding space. Emits promote/hold."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "/app/evaluate.py"],
        args=["--decision-out", decision, "--score-out", score],
    )


@dsl.container_component
def refresh(index: str):
    """Upsert the new item vectors into the LIVE index. STREAM_UPDATE only."""
    return dsl.ContainerSpec(
        image=IMAGE, command=["python", "/app/refresh_index.py"], args=[index]
    )


@dsl.container_component
def hold(score: float):
    """Gate closed. Fails the pipeline on purpose so the run shows red."""
    return dsl.ContainerSpec(
        image=IMAGE, command=["python", "/app/hold.py"], args=[score]
    )


@dsl.pipeline(
    name="reco-retrain",
    description="Retrain the two-tower; promote to the live index only if offline "
                "recall did not regress.",
)
def reco_retrain(index: str):
    t = train()
    t.set_env_variable("PROJECT", PROJECT)
    t.set_env_variable("TF_USE_LEGACY_KERAS", "1")
    t.set_cpu_limit("4").set_memory_limit("16G")
    t.set_caching_options(False)          # never serve a cached retrain. Ever. See the box.

    e = evaluate().after(t)
    e.set_env_variable("PROJECT", PROJECT)
    e.set_env_variable("RECALL_FLOOR", "0.35")
    e.set_env_variable("RECALL_MARGIN", "0.03")
    e.set_caching_options(False)

    # ── The gate ────────────────────────────────────────────────────────────
    with dsl.If(e.outputs["decision"] == "promote", name="recall-gate"):
        r = refresh(index=index)
        r.set_env_variable("PROJECT", PROJECT)
        r.set_caching_options(False)
    with dsl.Else(name="gate-closed"):
        h = hold(score=e.outputs["score"])
        h.set_env_variable("PROJECT", PROJECT)
        h.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(reco_retrain, "pipelines/reco_retrain.json")
    print("compiled → pipelines/reco_retrain.json")
```

> **⚠️ `set_caching_options(False)` on every task, and this one is a genuine silent failure.** Vertex Pipelines caches component executions by hashing their inputs. `train()` takes **no inputs at all** — so its cache key is constant, and on your *second* run Vertex will skip training entirely and hand the cached outputs straight to the evaluator. You get a **green pipeline that trained nothing**, evaluated last week's embeddings, and cheerfully re-promoted them. Nothing errors. Nothing warns. The only tell is a suspiciously fast run and a `Cached` badge on the node, which you will not be looking for. Caching is a fine default for deterministic transforms; it is exactly wrong for "retrain on whatever data arrived since yesterday."

### Step 15.3.5 — Submit it

**`pipelines/run_pipeline.py`**:

```python
import subprocess

from google.cloud import aiplatform

PROJECT, REGION = "reco-engine-adv", "europe-west3"
PIPELINE_ROOT = "gs://reco-engine-adv-artifacts/pipeline-root"
SA = "trainer@reco-engine-adv.iam.gserviceaccount.com"

# Resolve the live index by display name, so nothing is hard-coded.
index = subprocess.run(
    ["gcloud", "ai", "indexes", "list", f"--region={REGION}",
     "--filter=displayName=item-index", "--format=value(name)"],
    capture_output=True, text=True, check=True,
).stdout.strip()
if not index:
    raise SystemExit("no index named 'item-index' found — did Phase 8 run?")
print("index:", index)

aiplatform.init(project=PROJECT, location=REGION)

job = aiplatform.PipelineJob(
    display_name="reco-retrain",
    template_path="pipelines/reco_retrain.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"index": index},
    enable_caching=False,        # belt and braces, alongside the per-task setting
)
job.run(service_account=SA, sync=True)
print("pipeline finished:", job.state)
```

Run the whole thing:

```bash
cd ~/dev/reco-engine-adv
source .venv-kfp/bin/activate

python pipelines/reco_pipeline.py     # compile → pipelines/reco_retrain.json
python pipelines/run_pipeline.py      # submit → blocks until done
```

It prints a console URL; open it and watch `train → evaluate → recall-gate → refresh` light up as a real DAG. `pipeline finished: 4` is `PIPELINE_STATE_SUCCEEDED`. Expect 10–20 minutes, most of it provisioning.

> **The pipeline runs as `trainer@`, and needs no roles it doesn't already have** — Phase 1 gives it `aiplatform.user`, `bigquery.dataEditor`, `bigquery.jobUser`, `bigquery.readSessionUser`, `storage.objectAdmin`, `artifactregistry.reader`, and **`logging.logWriter`**. That last one was added in v6 *because of this step*: without it the container's stdout is discarded, and a failing task reports nothing but `The replica workerpool0-0 exited with a non-zero status of 1` over an empty log. If you built `infra/` from an older copy, add it and `terraform apply` **before** your first run — see the box in Phase 1.

> **Cost:** a pipeline run bills only while it runs (a few n1-standard machine-minutes — cents). Nothing here holds a node afterwards. The `reco-ops` image sits in Artifact Registry at fractions of a cent per month. There is no new off-switch to remember.

> **If `PipelineJob` rejects the pipeline root** with a location error, point `PIPELINE_ROOT` at `gs://reco-engine-adv-vectors/pipeline-root` — that bucket is single-region `europe-west3`. Same multi-region-vs-regional split that bit you in Phase 8. (Vertex normally accepts the EU multi-region bucket, which is why it is the default here.)

### Step 15.3.6 — Prove the gate actually gates

**A gate that has only ever said *yes* is not a gate; it is a decoration.** Make it say no.

Set `RECALL_FLOOR` to an impossible `"0.99"` in `reco_pipeline.py`, then recompile and re-run:

```bash
source .venv-kfp/bin/activate
python pipelines/reco_pipeline.py && python pipelines/run_pipeline.py
```

The `recall-gate` branch should go **unexecuted**, `gate-closed` should run `hold.py`, and the pipeline should end **FAILED** with the message in the logs. **Screenshot that DAG** — a red pipeline that refused to ship a bad model is a far better portfolio artefact than a green one that shipped a good one.

Then set the floor back to `0.35`, recompile, and re-run. Do it in that order: the failed run left new user vectors in BigQuery with no matching index upsert, so you are *inside the consistency window* until the successful run closes it.

### Reading the logs when a task fails

The DAG error names the failed task and nothing else:

```
The DAG failed because some tasks failed. The failed tasks are: [train].
```

The actual traceback is your container's stdout. Pull it (and note that container output often lands in `jsonPayload.message`, not `textPayload`):

```bash
gcloud logging read \
  'resource.type="aiplatform.googleapis.com/PipelineJob"
   AND resource.labels.pipeline_job_id="<PIPELINE_JOB_ID>   # the job id printed above"' \
  --project=reco-engine-adv --freshness=2h --limit=300 --order=asc \
  --format="value(textPayload,jsonPayload.message)"
```

**If that returns only four service-level lines** (`Waiting for job to be provisioned` / `Job is running` / the exit notice / `Job failed`) and none of your own `print()`s, **stop debugging the query — the SA is missing `roles/logging.logWriter`** and your logs are being discarded, not hidden. See Phase 1.

A clean **exit status 1** with no traceback usually means `train.py` hit its own `sys.exit()` guard — most often the 5,000-interaction-pair floor. Check the data before you check the code:

```bash
bq query --use_legacy_sql=false '
SELECT COUNT(*) AS pairs FROM `reco-engine-adv.reco.events`
WHERE event_type IN ("click","add_to_cart","purchase")'
```

(Exit **137**, by contrast, is an OOM kill — raise `set_memory_limit`.)

✅ **Checkpoint 15:** after a retrain + upsert, the API serves recommendations from fresh embeddings **with no redeploy**, and `test_retrieval.py` still beats the 20 % baseline. **And (15.3):** the pipeline runs green end-to-end, writes `gs://reco-engine-adv-artifacts/eval/champion.json`, and — with the floor rigged to 0.99 — closes the gate, skips the upsert, and fails the run.

> **Sanity-check the two metrics against each other, once.** `evaluate.py` (brute force, off-index) and `test_retrieval.py` (ANN, on-index) should agree closely after a successful promote, because at 500 items with `leaf_nodes_to_search_percent=100` the index *is* doing an exact scan. If the brute-force score is high and the index score is near chance, the upsert didn't land and you are in the consistency window. If they disagree in any other way, one of them is lying and it's worth finding out which before either number goes in your README.
>
> **And don't be alarmed by a score near 100 %.** A converged two-tower on this data will happily post a **99 %+** preferred-category share, and the first instinct — *"that's too good, something's leaking"* — is wrong here. Nothing leaks: `users_truth` is never a training input, and the metric is measured against it. The number is high because the *data* is easy. The generator plants a **70 % category preference**, the embedding space separates cleanly into five clusters, and an exact dot-product scan over 500 items then retrieves from the right cluster essentially every time. (Simulated: an aligned user scores **100 %**, a random one **18.6 %** against the same item space — the metric discriminates exactly as it should.) So the score is *correct*, and it is also *not impressive*, and your README must say the second part: quote it as **"99 % vs a 20 % random baseline **on synthetic data with a planted preference signal**."** Quoting it bare invites the reader to assume you leaked your labels — which is a worse conversation than the one you avoid by adding six words.

---

# Phase 16 — dbt experiment marts + dashboard (~2 hrs)

**Goal:** Turn raw impressions and feedback into tested, documented analytics tables — the daily job of an analytics engineer.

> **Term — dbt:** a tool that builds BigQuery tables/views out of `SELECT` statements, with dependency graphs (`ref()`), automated tests, and generated docs. The point isn't the SQL — it's that the SQL is **versioned, tested, and reproducible**.

dbt authenticates through your **Application Default Credentials**, so make sure this has been run at least once (it's the same login from Phase 0):

```bash
gcloud auth application-default login
```

Scaffold the project:

```bash
cd ~/dev/reco-engine-adv && source .venv/bin/activate
pip install dbt-bigquery
dbt init reco_dbt
```

`dbt init` asks a short series of questions — answer them exactly like this:

| Prompt | Answer |
|--------|--------|
| Which database would you like to use? | type the number next to **bigquery** |
| Desired authentication method | **oauth** |
| project (GCP project id) | **reco-engine-adv** |
| dataset (BigQuery dataset) | **reco** |
| threads | **4** |
| job_execution_timeout_seconds | press **Enter** (default 300) |
| Desired location | **EU** (must match your BigQuery multi-region) |

dbt writes these into `~/.dbt/profiles.yml` (on your machine, not in the repo — never commit credentials). Confirm the connection, then delete the sample models dbt scaffolds:

```bash
cd reco_dbt
dbt debug            # expect: "All checks passed!"
rm -r models/example
```

> **Then delete the matching block from `dbt_project.yml`.** `dbt init` scaffolds a config stanza that points at the folder you just removed:
> ```yaml
> models:
>   reco_dbt:
>     example:
>       +materialized: view
> ```
> Leave it and every `dbt build` prints `WARNING: Configuration paths exist in your dbt_project.yml file which do not apply to any resources: models.reco_dbt.example`. It's only a warning, but a repo that ships with a permanent warning in its build output is a repo nobody trusts the green ticks in. Delete the `example:` lines.

**`models/stg_impressions.sql`**

```sql
SELECT impression_id, user_id, variant, served_by, served_at
FROM {{ source('reco', 'impressions') }}
```

**`models/mart_experiment.sql`**

```sql
WITH clicks AS (
  SELECT DISTINCT impression_id FROM {{ source('reco', 'feedback') }} WHERE event = 'click'
),
convs AS (
  SELECT DISTINCT impression_id FROM {{ source('reco', 'feedback') }} WHERE event = 'conversion'
)
SELECT
  DATE(i.served_at) AS day,
  i.variant,
  COUNT(*) AS impressions,
  COUNTIF(i.served_by = 'popular') AS fallbacks,
  COUNTIF(c.impression_id IS NOT NULL) AS clicks,
  SAFE_DIVIDE(COUNTIF(c.impression_id IS NOT NULL), COUNT(*)) AS ctr,
  COUNTIF(v.impression_id IS NOT NULL) AS conversions,
  SAFE_DIVIDE(COUNTIF(v.impression_id IS NOT NULL), COUNT(*)) AS cvr
FROM {{ ref('stg_impressions') }} i
LEFT JOIN clicks c USING (impression_id)
LEFT JOIN convs  v USING (impression_id)
GROUP BY day, variant
```

**`models/sources.yml`**

```yaml
version: 2

sources:
  - name: reco
    database: reco-engine-adv
    schema: reco
    tables:
      - name: impressions
      - name: feedback

models:
  - name: stg_impressions
    columns:
      - name: impression_id
        tests:
          - not_null
          - unique
      - name: variant
        tests:
          - accepted_values:
              values: ['control', 'treatment']
      - name: served_by
        tests:
          - accepted_values:
              values: ['personalized', 'popular']
```

Run it from inside `reco_dbt/` with the venv active (`dbt build` compiles the models, creates the tables, and runs the tests):

```bash
dbt build            # from ~/dev/reco-engine-adv/reco_dbt
```

> **If the `unique` test on `impression_id` fails**, you have duplicate impressions — which at this scale means the API was double-logging or you replayed a simulator run against a table you'd already loaded. Track it down; don't delete the test. A failing test that you *understand* is worth more than a green one you rigged.

Then the dashboard: **Looker Studio → Create → Data source → BigQuery → reco-engine-adv → reco → `mart_experiment`**. Build:
- **CTR by variant over time** — two lines; the gap between them *is* your lift, and it's the chart that goes in the README;
- cumulative impressions per arm (proves the 50/50 split held);
- CVR by variant — the *business* story, sitting next to Phase 14's *model-health* story.

✅ **Checkpoint 16:** `dbt build` prints green with all tests passing, and the dashboard shows treatment above control.

---

# Phase 17 — CI/CD with GitHub Actions (keyless WIF) (~1.5 hrs)

**Goal:** Push to `main` → the API redeploys itself. No service-account key file anywhere.

> **Term — Workload Identity Federation (WIF):** instead of downloading a service-account JSON key and pasting it into GitHub secrets (where it lives forever, leaks eventually, and is the #1 finding in cloud security audits), GCP **trusts GitHub's OIDC token directly**. GitHub proves "this workflow is running in repo X on branch Y"; GCP checks that against a trust condition and mints a short-lived token. No long-lived secret exists to steal.

## Step 17.1 — Set up the trust relationship

```bash
gcloud services enable iam.googleapis.com sts.googleapis.com iamcredentials.googleapis.com \
  --project reco-engine-adv

PROJECT_ID=reco-engine-adv
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
REPO="<your-github-username>/reco-engine-adv"

gcloud iam workload-identity-pools create github \
  --project=$PROJECT_ID --location=global --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=$PROJECT_ID --location=global --workload-identity-pool=github \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${REPO}'"

gcloud iam service-accounts create github-deployer --project=$PROJECT_ID \
  --display-name="GitHub Actions Deployer"
DEPLOYER="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

for ROLE in roles/run.admin roles/iam.serviceAccountUser roles/cloudbuild.builds.editor \
            roles/artifactregistry.writer roles/storage.admin ; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${DEPLOYER}" --role="$ROLE"
done

gcloud iam service-accounts add-iam-policy-binding $DEPLOYER --project=$PROJECT_ID \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${REPO}"
```

## Step 17.2 — The workflow file

Written via heredoc — **never paste YAML into a shell prompt**, the indentation will not survive.

```bash
cd ~/dev/reco-engine-adv
mkdir -p .github/workflows

cat > .github/workflows/deploy.yml << 'EOF'
name: Deploy reco-engine
on:
  push:
    branches: [main]
permissions:
  contents: read
  id-token: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - id: auth
        uses: google-github-actions/auth@v3
        with:
          project_id: 'reco-engine-adv'
          workload_identity_provider: 'projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-provider'
          service_account: 'github-deployer@reco-engine-adv.iam.gserviceaccount.com'
      - uses: google-github-actions/setup-gcloud@v3
      - name: Deploy API
        run: |
          gcloud run deploy reco-api --source ./api --region europe-west3 \
            --service-account reco-api@reco-engine-adv.iam.gserviceaccount.com \
            --allow-unauthenticated --quiet
EOF

sed -i '' "s/PROJECT_NUMBER/$PROJECT_NUMBER/" .github/workflows/deploy.yml
grep workload_identity_provider .github/workflows/deploy.yml
```

> **`sed -i ''` is macOS (BSD) syntax** — the empty `''` is a required "no backup file" argument there. On Linux, drop it: `sed -i "s/.../.../" ...`. The `grep` line must now print your real project **number** in the path, not the literal string `PROJECT_NUMBER`. If it doesn't, the auth step will fail with a resource-not-found error that looks like a permissions problem and isn't.

> **`project_id:` on the auth step is not decoration.** The `workload_identity_provider` path carries the project *number*, and the action cannot turn a number back into an ID without extra permissions — so it doesn't try. Omit `project_id` and the first `gcloud` command in the next step dies with `The required property [project] is not currently set`, which reads like a broken login and isn't one.

> **No `--set-env-vars` in the workflow — deliberately.** `gcloud run deploy` **preserves** existing environment variables when you don't pass that flag, so `INDEX_ENDPOINT` and friends survive every redeploy. If you *do* pass `--set-env-vars`, it **replaces the whole set** — and your API comes back up with no `INDEX_ENDPOINT` and crashes on the first request with a `KeyError`. (Use `--update-env-vars` if you ever need to change just one.)

> **This workflow deploys the API only** — not your infrastructure. That's deliberate: Terraform changes are rare, and you'd rather run `terraform apply` yourself than have every push silently touch IAM and datasets.

## Step 17.3 — Push

```bash
cd ~/dev/reco-engine-adv

cat > .gitignore << 'EOF'
.venv/
.venv-beam/
.venv-kfp/
__pycache__/
*.pyc
infra/.terraform/
*.tfstate
*.tfstate.*
*.tfvars
*-key.json
generator/catalog.jsonl
generator/users.jsonl
logs/
.DS_Store
EOF

git init && git branch -M main && git add .
git status                       # read this properly — see the box below
git commit -m "real-time recommender with A/B testing"
gh auth login
gh repo create reco-engine-adv --public --source=. --remote=origin --push
```

> **⚠️ Read `git status` before you commit, and read it for what *shouldn't* be there.** Three of the four venv-shaped landmines are covered above; the one that gets people is **`.venv-kfp/`**, because it is created much later (Step 15.3) than the other two, in a step whose subject was pipelines and not hygiene. Commit it and you push several thousand files of compiled dependencies into a public repo — GitHub will cheerfully report your project as part C, part PowerShell, and the venv's `bin/activate` will publish the absolute path of your home directory.
>
> Verify, rather than trusting the list:
>
> ```bash
> git check-ignore -v .venv-kfp logs        # must print a matching rule for each
> git status --short | grep -E "venv|tfstate|\.json$" # must print nothing
> ```
>
> If you already committed one: `git rm -r --cached .venv-kfp`, fix `.gitignore`, then `git commit --amend` and force-push **before** anyone forks you. A file removed in a later commit is still in the history.

> **If `gh repo create` says the repo already exists**, skip it and just wire up the remote:
> ```bash
> git remote add origin https://github.com/<your-github-username>/reco-engine-adv.git
> git push -u origin main
> ```

**Watch the deploy run.** The push triggers the workflow immediately — on GitHub, open the repo → **Actions** tab → the **Deploy reco-engine** run. Yellow (running) → green (done); click in to read step logs if it fails. First runs commonly fail on a role the `github-deployer` SA is missing — the log names the exact permission, which you add with one more `gcloud projects add-iam-policy-binding` line.

> Reminder: the WIF trust condition is bound to the **exact** repo path. Renaming the repo breaks CI/CD until you update the provider's `--attribute-condition`.

✅ **Checkpoint 17:** a push shows a green **Deploy reco-engine** run in the Actions tab, and the live API reflects your change.

---

# Phase 18 — Package it as a portfolio piece (~2 hrs)

**Goal:** Make a stranger understand what you built, and believe the numbers, in one screen.

- **README headline = the A/B result:** *"Personalized recommendations lifted CTR X % over a popular-items baseline (absolute difference d, 95 % CI [lo, hi], z = …, p < …)."* Immediately under it, the offline metric: *"preferred-category share in top-10: Y % vs a 20 % random baseline."* Two numbers, both checkable, both honest.
  > **State the caveats in the README, not just in conversation.** The fallback rate, the peeking caveat, the fact that the simulator's click model is *your* assumption about user behaviour and not a measured one. A project that names its own limitations is one you can trust the rest of; a project that doesn't is one you have to go looking through.
- **Architecture diagram** + one sentence per component.
- **The feature-store paragraph** — training-serving skew, point-in-time joins, the append-only snapshot design, and the BigQuery-vs-Bigtable economics with the per-node-hour floor called out explicitly. Quote the Phase 6.2 measurement as a **pair, with the measurement point named**, never as one number: *"cold read 216 ms, warm median 103 ms over 20 reads from a laptop outside GCP."* A lone cold read says Bigtable is slow **and** expensive and collapses your own argument; a lone median invites "over how many reads, from where?". The pair answers both before they're asked.
- **The retrieve→rank paragraph** — why two stages, what each optimizes for.
- **The least-privilege paragraph** *(short, and worth more than its length)* — every component runs as its own service account with an explicit role list, and the training job proved why that's non-trivial: `train.py` passed on the laptop and 403'd on Vertex, because `.to_dataframe()` silently reaches for the **BigQuery Storage Read API**, which is a *third* permission that neither `jobUser` nor `dataEditor` grants. "Works as me, fails as the robot" is the single most common shape of GCP incident, and having *hit* it, diagnosed it from the job's logs, and pinned the fix in Terraform is a better story than any role list.
- **The experiment-design paragraph** — per-user hash randomization, append-only feedback (and *why*: the streaming buffer), intent-to-treat, the fallback tagging.
- **The gated-retraining paragraph** *(new, and it earns its place)* — the retraining pipeline evaluates the new embeddings **off-index**, by brute force, and only upserts into the live Vector Search index if offline recall hasn't regressed against a stored champion score. Say *why* the obvious design fails: scoring the new model against the live index right after a retrain measures the **consistency window** (new user vectors vs. old item vectors), not the model, and returns chance every time — so a gate wired to it can never open. Then name what the gate does **not** cover: user vectors are published to BigQuery unconditionally, so a *closed* gate still leaves the system mid-window. "Here is the gate, here is precisely what it doesn't protect, here is what would close it" is a description of a system someone has actually operated. A gate presented as airtight is a gate nobody has tested.
- **Repo layout:** `infra/`, `generator/`, `dataflow/`, `featurestore/`, `training/`, `pipelines/`, `api/`, `analysis/`, `reco_dbt/`, `.github/workflows/`.
- **"What I'd do next"** — a real-time user tower on an endpoint (session freshness); true point-in-time feature retrieval for training; sequence models (transformers) for session-based recs; multi-armed bandits instead of a fixed 50/50 split; Bigtable online serving at real QPS; **Vector Search index aliasing for blue/green swaps** and **staging user vectors behind the recall gate** (the two halves of the Phase 15 consistency window — the retraining pipeline currently gates the index but not the user-embedding table, so a rejected model still lands half its output in production); scheduling the retraining pipeline on Cloud Scheduler instead of running it by hand.

**The one-sentence summary:**
> "I built a real-time recommender on GCP — interactions stream through Dataflow into append-only BigQuery feature tables registered in Vertex AI Feature Store; a two-tower TFRS model produces embeddings served from Vector Search in milliseconds; a boosted-tree ranker orders the candidates; a hash-randomized A/B test measured a statistically significant CTR lift over a popular-items baseline; and a Vertex AI Pipeline retrains on fresh events and upserts new embeddings into the live index only when a recall gate says they haven't regressed — all Terraform-provisioned and deployed via keyless GitHub Actions."

That one sentence hits streaming data engineering, feature stores, deep retrieval, ANN serving, two-stage ranking, experimentation, statistics, MLOps, IaC, and CI/CD.

✅ **Checkpoint 18:** a stranger reads the README and gets the architecture, the significant result, and the lifecycle story in one screen.

---

---

## Cost hygiene — the off-switches, after **every** session

Three things bill while switched on. Run these when you stop for the day.

**1. The Dataflow streaming job**

```bash
gcloud dataflow jobs list --region europe-west3 --status active
gcloud dataflow jobs drain <JOB_ID> --region europe-west3
```

*(If a `ZONE_RESOURCE_POOL_EXHAUSTED` stockout in Step 3.3 pushed you to launch in `europe-west1` or `europe-west4`, use **that** region in both commands. `jobs list` is region-scoped and will report an empty list for the region you didn't use.)*

**2. The Vector Search deployed index** (an idle deployed index still bills for its node)

```bash
INDEX_ENDPOINT_ID=$(gcloud ai index-endpoints list --region=europe-west3 \
  --filter='displayName=item-index-endpoint' --format='value(name)' | awk -F/ '{print $NF}')

gcloud ai index-endpoints undeploy-index "$INDEX_ENDPOINT_ID" \
  --deployed-index-id=items_v1 --region=europe-west3
```

Redeploying later takes ~20–30 minutes, so do it *before* you sit down to demo, not during:

```bash
INDEX_ID=$(gcloud ai indexes list --region=europe-west3 \
  --filter='displayName=item-index' --format='value(name)' | awk -F/ '{print $NF}')

gcloud ai index-endpoints deploy-index "$INDEX_ENDPOINT_ID" \
  --index="$INDEX_ID" --deployed-index-id=items_v1 \
  --display-name=items_v1 --region=europe-west3
```

**3. The Bigtable online store** — already deleted by the Phase 6.2 `finally` block. Verify:

```bash
python -c "import vertexai; vertexai.init(project='reco-engine-adv', location='europe-west3'); from vertexai.resources.preview import feature_store as fs; print(fs.FeatureOnlineStore.list())"
# must print []
```

Two commands that will **lie to you** here, and you must not substitute either one:
- `gcloud bigtable instances list` — shows an **empty list while the store is billing**. The Bigtable lives inside Google's project, not yours.
- `gcloud ai feature-online-stores list` — **doesn't exist in many gcloud versions** (`ERROR: (gcloud.ai) Invalid choice`). An erroring command is not a green light; it's a command that couldn't answer.

The Python one-liner above works regardless of SDK version. Use it. (Console path, if you prefer eyes on it: **Vertex AI → Feature Store → Online store**.)

Also stop the **generator** (Ctrl-C) — it's free, but it keeps filling `reco.events` and slowly skews your drift baselines.

## Full teardown

The Vector Search index and endpoint live **outside** Terraform, so `terraform destroy` won't touch them. Kill them first, then the substrate:

```bash
# 1) undeploy (above), then delete the endpoint and the index
gcloud ai index-endpoints delete "$INDEX_ENDPOINT_ID" --region=europe-west3
gcloud ai indexes delete "$INDEX_ID" --region=europe-west3

# 2) drain any Dataflow job (above)

# 3) everything Terraform provisioned — including the SQL-created tables, views
#    and the BQML model, thanks to delete_contents_on_destroy on the dataset,
#    and BOTH buckets, thanks to force_destroy on each
cd ~/dev/reco-engine-adv/infra && terraform destroy
```

Everything else — Cloud Run, Pub/Sub, BigQuery storage, BQML, one-shot training jobs — is scale-to-zero or pennies. Rebuilding takes about 30 minutes, and the index deploy is the long pole.

---

## Licence

MIT. See `LICENSE`.

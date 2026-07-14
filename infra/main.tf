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

resource "google_storage_bucket" "artifacts" {
  name                        = "reco-engine-adv-artifacts" # globally unique — suffix if taken
  location                    = "EU"
  uniform_bucket_level_access = true
  force_destroy               = true
  depends_on                  = [google_project_service.apis]
}
# Vector Search reads contents_delta_uri from a bucket in the SAME single region
# as the index (europe-west3). The `artifacts` bucket is EU multi-region, which
# Vector Search rejects — so item vectors get their own regional bucket.
resource "google_storage_bucket" "vectors" {
  name                        = "reco-engine-adv-vectors" # globally unique — suffix if taken
  location                    = "europe-west3"            # REGIONAL, not "EU"
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
# blocks DML on them for ~90 minutes — a lesson already learned the hard
# way in the hate-speech build.)
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
locals {
  df_roles = [
    "roles/dataflow.worker",
    "roles/pubsub.subscriber",
    "roles/pubsub.viewer",
    "roles/bigquery.dataEditor",
    "roles/storage.objectAdmin",
  ]
  trainer_roles = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/bigquery.readSessionUser",
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",   # without this, container stdout is DISCARDED and
                                 # every failure looks like a bare "exit status 1"
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
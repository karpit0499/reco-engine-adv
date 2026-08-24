# Recommender API contract

`/recommend` and `/feedback` trigger billable cloud work or mutate experiment data. They fail
closed unless `API_BEARER_TOKEN` is configured (32–512 non-whitespace characters) and the
caller supplies it as `Authorization: Bearer …`. The GCP workflow additionally deploys Cloud
Run without public invocation. When Cloud Run IAM is enabled, callers can put the Google
identity token in `X-Serverless-Authorization` and leave `Authorization` for the application
token.

Requests are JSON-only and capped at 2 KiB. IDs are bounded; idempotency IDs must be canonical,
non-zero UUIDs. Extra fields are rejected.

```json
POST /recommend
{"user_id":"u0001","request_id":"123e4567-e89b-42d3-a456-426614174000"}
```

The request UUID becomes the impression UUID and the BigQuery streaming insert ID. If logging
the impression fails, the API does not return recommendations because doing so would create an
unmeasured exposure.

```json
POST /feedback
{
  "user_id":"u0001",
  "impression_id":"123e4567-e89b-42d3-a456-426614174000",
  "event_id":"e35ea608-05a0-48e6-9b55-f895600cf23c",
  "event":"click"
}
```

Feedback is accepted only when the impression belongs to the same user. The event UUID is used
as the BigQuery streaming insert ID for best-effort replay deduplication. BigQuery does not
provide a uniqueness constraint here, so downstream experiment marts still deduplicate by
impression as defense in depth.

The deployment workflow is manual and requires the literal confirmation `DEPLOY`. It expects a
Secret Manager version named `reco-api-bearer`; creating that secret and changing GCP IAM,
quotas, or resources are deliberately outside this repository-only workflow.

The `gcp-production` GitHub environment must also define `INDEX_RESOURCE_NAME`,
`INDEX_ENDPOINT`, and `DEPLOYED_INDEX_ID`. The first two are fully qualified Vertex resource
names. Each treatment request verifies that the active-generation record has that exact binding,
captures its generation with the user vector, and queries Vertex with the corresponding
`generation` namespace filter. Missing or mismatched binding fails closed.

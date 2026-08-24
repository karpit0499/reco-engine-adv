# Security policy

Please report vulnerabilities privately through this repository's **Security** tab by
opening a private vulnerability report. Do not include exploit details in a public issue.

The static demo is synthetic and does not call the serving API. Production endpoints must
remain authenticated; the GCP deployment workflow is deliberately manual so infrastructure
owners can review identity, quotas, IAM, and cost before rollout.

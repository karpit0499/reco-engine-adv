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
    enable_caching=False,        # belt and braces alongside the per-task setting
)
job.run(service_account=SA, sync=True)
print("pipeline finished:", job.state)
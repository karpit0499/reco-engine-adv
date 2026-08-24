import os
import uuid
from datetime import datetime, timezone

from google.cloud import aiplatform
from generation_contract import validate_ops_image_uri, validate_source_revision

PROJECT, REGION = "reco-engine-adv", "europe-west3"
BUCKET = "gs://reco-engine-adv-artifacts"
IMAGE = validate_ops_image_uri(os.environ["RECO_OPS_IMAGE"])
GENERATION_ID = os.environ.get(
    "GENERATION_ID",
    f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}",
)
INDEX_RESOURCE_NAME = os.environ["INDEX_RESOURCE_NAME"]
INDEX_ENDPOINT = os.environ["INDEX_ENDPOINT"]
DEPLOYED_INDEX_ID = os.environ["DEPLOYED_INDEX_ID"]
SOURCE_REVISION = validate_source_revision(os.environ["SOURCE_REVISION"])

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
        "AIP_MODEL_DIR": f"{BUCKET}/models/twotower",
        "GENERATION_ID": GENERATION_ID,
        "INDEX_RESOURCE_NAME": INDEX_RESOURCE_NAME,
        "INDEX_ENDPOINT": INDEX_ENDPOINT,
        "DEPLOYED_INDEX_ID": DEPLOYED_INDEX_ID,
        "RECO_OPS_IMAGE": IMAGE,
        "SOURCE_REVISION": SOURCE_REVISION,
        "TF_USE_LEGACY_KERAS": "1",
    },
    service_account="trainer@reco-engine-adv.iam.gserviceaccount.com",
    sync=True,                          # block until the job finishes
)
print("staged training generation:", GENERATION_ID)

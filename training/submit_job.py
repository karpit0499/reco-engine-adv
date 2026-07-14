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
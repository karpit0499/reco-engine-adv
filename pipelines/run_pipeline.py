"""Compile and submit one clean, digest-pinned recommender pipeline."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone

from google.cloud import aiplatform


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from generation_contract import (  # noqa: E402
    validate_binding,
    validate_ops_image_uri,
    validate_source_revision,
)


PROJECT, REGION = "reco-engine-adv", "europe-west3"
PIPELINE_ROOT = "gs://reco-engine-adv-artifacts/pipeline-root"
SA = "trainer@reco-engine-adv.iam.gserviceaccount.com"
RESERVED_IMAGE_DIGEST = "0" * 64
RESERVED_SOURCE_REVISION = "0" * 40


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_submission_provenance() -> tuple[str, str]:
    image = validate_ops_image_uri(os.environ.get("RECO_OPS_IMAGE"))
    source_revision = validate_source_revision(os.environ.get("SOURCE_REVISION"))
    if image.endswith(f"@sha256:{RESERVED_IMAGE_DIGEST}") or image.startswith(
        "registry.invalid/"
    ):
        raise RuntimeError("the deterministic CI image placeholder cannot be submitted")
    if source_revision == RESERVED_SOURCE_REVISION:
        raise RuntimeError("the deterministic CI source placeholder cannot be submitted")
    head = validate_source_revision(git_output("rev-parse", "HEAD"))
    if source_revision != head:
        raise RuntimeError("SOURCE_REVISION must equal the exact checked-out Git commit")
    if git_output("status", "--porcelain"):
        raise RuntimeError("pipeline submission requires a completely clean worktree")
    return image, source_revision


def compile_exact_spec(image: str, source_revision: str, output: pathlib.Path) -> dict:
    environment = os.environ.copy()
    environment["RECO_OPS_IMAGE"] = image
    environment["SOURCE_REVISION"] = source_revision
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipelines/reco_pipeline.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    executors = document.get("deploymentSpec", {}).get("executors", {})
    if not executors:
        raise RuntimeError("compiled pipeline has no executors")
    for name, executor in executors.items():
        container = executor.get("container", {})
        if container.get("image") != image:
            raise RuntimeError(f"executor {name} is not bound to the exact ops image")
    serialized = output.read_text(encoding="utf-8")
    if serialized.count(source_revision) < 3 or serialized.count(image) < len(executors):
        raise RuntimeError("compiled pipeline does not carry complete source/image provenance")
    return document


def main() -> None:
    image, source_revision = validate_submission_provenance()
    binding = validate_binding(
        os.environ["INDEX_RESOURCE_NAME"],
        os.environ["INDEX_ENDPOINT"],
        os.environ["DEPLOYED_INDEX_ID"],
    )
    generation_id = (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    )
    print("index:", binding.index_name)
    print("index endpoint:", binding.index_endpoint)
    print("deployed index id:", binding.deployed_index_id)
    print("ops image:", image)
    print("source revision:", source_revision)
    print("generation:", generation_id)

    with tempfile.TemporaryDirectory(prefix="reco-pipeline-") as directory:
        template = pathlib.Path(directory) / "reco_retrain.json"
        compile_exact_spec(image, source_revision, template)
        aiplatform.init(project=PROJECT, location=REGION)
        job = aiplatform.PipelineJob(
            display_name="reco-retrain",
            template_path=str(template),
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "generation_id": generation_id,
                "index_name": binding.index_name,
                "index_endpoint": binding.index_endpoint,
                "deployed_index_id": binding.deployed_index_id,
            },
            enable_caching=False,
        )
        job.run(service_account=SA, sync=True)
    print("pipeline finished:", job.state)


if __name__ == "__main__":
    main()

"""Compile the digest- and source-pinned recommender promotion pipeline."""

import argparse
import os
import pathlib
import sys

from kfp import compiler, dsl


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from generation_contract import (  # noqa: E402
    validate_ops_image_uri,
    validate_source_revision,
)


PROJECT, REGION = "reco-engine-adv", "europe-west3"
# KFP component images are compile-time values. Requiring both here makes one
# immutable image/source pair visible in every executor and prevents a task from
# claiming provenance that differs from the bytes it actually ran.
IMAGE = validate_ops_image_uri(os.environ.get("RECO_OPS_IMAGE"))
SOURCE_REVISION = validate_source_revision(os.environ.get("SOURCE_REVISION"))


@dsl.container_component
def train(
    generation_id: str,
    index_name: str,
    index_endpoint: str,
    deployed_index_id: str,
):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "/app/train.py"],
        args=[
            "--generation-id", generation_id,
            "--index-name", index_name,
            "--index-endpoint", index_endpoint,
            "--deployed-index-id", deployed_index_id,
            "--ops-image-uri", IMAGE,
            "--source-revision", SOURCE_REVISION,
        ],
    )


@dsl.container_component
def evaluate(
    generation_id: str,
    index_name: str,
    index_endpoint: str,
    deployed_index_id: str,
    decision: dsl.OutputPath(str),
    score: dsl.OutputPath(float),
    active_generation: dsl.OutputPath(str),
    active_epoch: dsl.OutputPath(int),
    manifest_generation: dsl.OutputPath(int),
    manifest_sha256: dsl.OutputPath(str),
    active_manifest_generation: dsl.OutputPath(int),
    active_manifest_sha256: dsl.OutputPath(str),
    active_ops_image_uri: dsl.OutputPath(str),
    active_source_revision: dsl.OutputPath(str),
    active_champion_score: dsl.OutputPath(float),
):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "/app/evaluate.py"],
        args=[
            "--generation-id", generation_id,
            "--index-name", index_name,
            "--index-endpoint", index_endpoint,
            "--deployed-index-id", deployed_index_id,
            "--ops-image-uri", IMAGE,
            "--source-revision", SOURCE_REVISION,
            "--decision-out", decision,
            "--score-out", score,
            "--active-generation-out", active_generation,
            "--active-epoch-out", active_epoch,
            "--manifest-generation-out", manifest_generation,
            "--manifest-sha256-out", manifest_sha256,
            "--active-manifest-generation-out", active_manifest_generation,
            "--active-manifest-sha256-out", active_manifest_sha256,
            "--active-ops-image-uri-out", active_ops_image_uri,
            "--active-source-revision-out", active_source_revision,
            "--active-champion-score-out", active_champion_score,
        ],
    )


@dsl.container_component
def promote(
    generation_id: str,
    score: float,
    evaluated_active_generation: str,
    evaluated_active_epoch: int,
    evaluated_manifest_generation: int,
    evaluated_manifest_sha256: str,
    evaluated_active_manifest_generation: int,
    evaluated_active_manifest_sha256: str,
    evaluated_active_ops_image_uri: str,
    evaluated_active_source_revision: str,
    evaluated_active_champion_score: float,
    index_name: str,
    index_endpoint: str,
    deployed_index_id: str,
):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "/app/promote.py"],
        args=[
            "--generation-id", generation_id,
            "--score", score,
            "--evaluated-active-generation", evaluated_active_generation,
            "--evaluated-active-epoch", evaluated_active_epoch,
            "--evaluated-manifest-generation", evaluated_manifest_generation,
            "--evaluated-manifest-sha256", evaluated_manifest_sha256,
            "--evaluated-ops-image-uri", IMAGE,
            "--evaluated-source-revision", SOURCE_REVISION,
            "--evaluated-active-manifest-generation", evaluated_active_manifest_generation,
            "--evaluated-active-manifest-sha256", evaluated_active_manifest_sha256,
            "--evaluated-active-ops-image-uri", evaluated_active_ops_image_uri,
            "--evaluated-active-source-revision", evaluated_active_source_revision,
            "--evaluated-active-champion-score", evaluated_active_champion_score,
            "--index-name", index_name,
            "--index-endpoint", index_endpoint,
            "--deployed-index-id", deployed_index_id,
        ],
    )


@dsl.container_component
def hold(score: float, generation_id: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "/app/hold.py"],
        args=[score, generation_id],
    )


@dsl.pipeline(
    name="reco-retrain",
    description=(
        "Generation-filtered, receipt-bound and fenced promotion saga for one "
        "immutable user/item embedding generation."
    ),
)
def reco_retrain(
    generation_id: str,
    index_name: str,
    index_endpoint: str,
    deployed_index_id: str,
):
    training_task = train(
        generation_id=generation_id,
        index_name=index_name,
        index_endpoint=index_endpoint,
        deployed_index_id=deployed_index_id,
    )
    training_task.set_env_variable("PROJECT", PROJECT)
    training_task.set_env_variable("TF_USE_LEGACY_KERAS", "1")
    training_task.set_cpu_limit("4").set_memory_limit("16G")
    training_task.set_caching_options(False)

    evaluation_task = evaluate(
        generation_id=generation_id,
        index_name=index_name,
        index_endpoint=index_endpoint,
        deployed_index_id=deployed_index_id,
    ).after(training_task)
    evaluation_task.set_env_variable("PROJECT", PROJECT)
    evaluation_task.set_env_variable("RECALL_FLOOR", "0.35")
    evaluation_task.set_env_variable("RECALL_MARGIN", "0.03")
    evaluation_task.set_caching_options(False)

    with dsl.If(evaluation_task.outputs["decision"] == "promote", name="recall-gate"):
        promotion_task = promote(
            generation_id=generation_id,
            score=evaluation_task.outputs["score"],
            evaluated_active_generation=evaluation_task.outputs["active_generation"],
            evaluated_active_epoch=evaluation_task.outputs["active_epoch"],
            evaluated_manifest_generation=evaluation_task.outputs["manifest_generation"],
            evaluated_manifest_sha256=evaluation_task.outputs["manifest_sha256"],
            evaluated_active_manifest_generation=evaluation_task.outputs[
                "active_manifest_generation"
            ],
            evaluated_active_manifest_sha256=evaluation_task.outputs[
                "active_manifest_sha256"
            ],
            evaluated_active_ops_image_uri=evaluation_task.outputs[
                "active_ops_image_uri"
            ],
            evaluated_active_source_revision=evaluation_task.outputs[
                "active_source_revision"
            ],
            evaluated_active_champion_score=evaluation_task.outputs[
                "active_champion_score"
            ],
            index_name=index_name,
            index_endpoint=index_endpoint,
            deployed_index_id=deployed_index_id,
        )
        promotion_task.set_env_variable("PROJECT", PROJECT)
        promotion_task.set_env_variable("REGION", REGION)
        promotion_task.set_caching_options(False)
    with dsl.Else(name="gate-closed"):
        hold_task = hold(
            score=evaluation_task.outputs["score"], generation_id=generation_id
        )
        hold_task.set_env_variable("PROJECT", PROJECT)
        hold_task.set_caching_options(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "pipelines/reco_retrain.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(reco_retrain, str(args.output))
    print(f"compiled -> {args.output}")


if __name__ == "__main__":
    main()

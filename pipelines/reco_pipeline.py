from kfp import compiler, dsl

PROJECT, REGION = "reco-engine-adv", "europe-west3"
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT}/reco-training/reco-ops:v1"


@dsl.container_component
def train():
    """Retrain the two-tower. Writes: towers → GCS, items.json → both buckets,
    user vectors → BigQuery (appended with a new trained_at)."""
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
    """Gate closed. Fails the pipeline on purpose so it shows red."""
    return dsl.ContainerSpec(
        image=IMAGE, command=["python", "/app/hold.py"], args=[score]
    )


@dsl.pipeline(
    name="reco-retrain",
    description="Retrain the two-tower and promote to the live index only if "
                "offline recall did not regress.",
)
def reco_retrain(index: str):
    t = train()
    t.set_env_variable("PROJECT", PROJECT)
    t.set_env_variable("TF_USE_LEGACY_KERAS", "1")
    t.set_cpu_limit("4").set_memory_limit("16G")
    t.set_caching_options(False)          # never serve a cached retrain. Ever.

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
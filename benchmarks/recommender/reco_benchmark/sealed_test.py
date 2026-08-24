from __future__ import annotations

from pathlib import Path

from .dataset import EvaluationData, load_prepared_split


def load_sealed_test_data(
    artifact: Path, manifest: dict, protocol: dict
) -> EvaluationData:
    """Load the separately prepared test artifact after authorization."""

    return load_prepared_split(artifact, manifest, protocol, expected_stage="test")

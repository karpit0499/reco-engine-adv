from __future__ import annotations

from pathlib import Path

from .dataset import EvaluationData, load_prepared_split


def load_validation_data(
    artifact: Path, manifest: dict, protocol: dict
) -> EvaluationData:
    """Load the validation-only prepared artifact without opening raw or test data."""

    return load_prepared_split(artifact, manifest, protocol, expected_stage="validation")

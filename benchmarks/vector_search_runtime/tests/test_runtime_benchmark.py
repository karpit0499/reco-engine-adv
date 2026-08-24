from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmarks.vector_search_runtime.runtime_benchmark import (
    BenchmarkConfig,
    BenchmarkContractError,
    CloudClients,
    GENERATION_NAMESPACE,
    build_plan,
    capture_control_snapshot,
    exact_top_k,
    main,
    parse_artifact,
    parse_live_promotion_manifest,
    read_active_pointer,
    read_immutable_manifest,
    read_live_vertex_contract,
    run_benchmark,
    validate_report,
    write_json_atomic,
)
from training.generation_contract import (
    GENERATION_NAMESPACE as PRODUCTION_NAMESPACE,
    canonical_item_payload,
)


INDEX = "projects/project-a/locations/europe-west3/indexes/index-a"
ENDPOINT = "projects/project-a/locations/europe-west3/indexEndpoints/endpoint-a"
DEPLOYED = "items_blue"
GENERATION = "gen-20260820"
PREVIOUS = "gen-20260819"
TABLE = "project-a.reco.active_embedding_generation"
OPS_IMAGE = (
    "europe-docker.pkg.dev/project-a/reco/ops@sha256:" + "a" * 64
)
SOURCE_REVISION = "b" * 40
MANIFEST_GENERATION = 123
ITEM_GENERATION = 456


def catalog_rows() -> list[dict]:
    return [
        {"id": f"item-{position:03d}", "embedding": [float(position), 1.0]}
        for position in range(100)
    ]


def item_digest() -> str:
    payload, _ = canonical_item_payload(catalog_rows(), dimensions=2)
    return hashlib.sha256(payload).hexdigest()


def manifest_document() -> dict:
    return {
        "schema_version": 3,
        "generation_id": GENERATION,
        "created_at": "2026-08-20T11:00:00+00:00",
        "trained_at": "2026-08-20T10:00:00+00:00",
        "dimensions": 2,
        "serving": {
            "index_name": INDEX,
            "index_endpoint": ENDPOINT,
            "deployed_index_id": DEPLOYED,
            "generation_namespace": "generation",
        },
        "provenance": {
            "ops_image_uri": OPS_IMAGE,
            "source_revision": SOURCE_REVISION,
        },
        "items": {
            "count": 100,
            "sha256": item_digest(),
            "canonical_uri": (
                f"gs://project-a-artifacts/candidate-generations/{GENERATION}/items.json"
            ),
            "canonical_object_generation": ITEM_GENERATION,
            "regional_uri": (
                f"gs://project-a-vectors/candidate-generations/{GENERATION}/items.json"
            ),
            "regional_object_generation": 457,
            "canary_item_id": "item-099",
            "canary_datapoint_id": f"{GENERATION}::item-099",
        },
        "users": {
            "count": 25,
            "dimensions": 2,
            "sha256": "d" * 64,
            "uri": (
                f"gs://project-a-artifacts/candidate-generations/{GENERATION}/users.json"
            ),
            "object_generation": 458,
            "table": "project-a.reco.user_embedding_generations",
            "generation_id": GENERATION,
            "canary_user_id": "private-canary-user",
        },
        "models": {
            "user_tower_uri": f"gs://project-a-artifacts/models/{GENERATION}/user_tower",
            "item_tower_uri": f"gs://project-a-artifacts/models/{GENERATION}/item_tower",
        },
    }


def manifest_payload() -> bytes:
    return json.dumps(
        manifest_document(), sort_keys=True, separators=(",", ":")
    ).encode()


def artifact_document(*, synthetic: bool = False) -> dict:
    payload = manifest_payload()
    return {
        "schema_version": 2,
        "artifact_kind": "production_generation_embedding_snapshot",
        "created_at": "2026-08-20T12:00:00Z",
        "source": {
            "kind": "production_interaction_embedding_export",
            "contains_synthetic_data": synthetic,
            "description": "pseudonymous interaction-trained active embeddings",
            "interaction_count": 10_000,
            "provenance_verification": "operator_attested",
        },
        "expected_active_pointer": {
            "table": TABLE,
            "generation_id": GENERATION,
            "epoch": 7,
            "fence_epoch": 7,
            "previous_generation_id": PREVIOUS,
            "manifest": {
                "uri": (
                    f"gs://project-a-artifacts/candidate-generations/{GENERATION}/manifest.json"
                ),
                "object_generation": MANIFEST_GENERATION,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "provenance": {
                "ops_image_uri": OPS_IMAGE,
                "source_revision": SOURCE_REVISION,
            },
            "retained_generation_ids": [GENERATION, PREVIOUS],
            "champion_score": 0.72,
            "serving": {
                "index_resource": INDEX,
                "index_endpoint": ENDPOINT,
                "deployed_index_id": DEPLOYED,
            },
            "promoted_at": "2026-08-20T11:30:00Z",
        },
        "manifest_contract": {
            "schema_version": 3,
            "dimensions": 2,
            "items": {
                "count": 100,
                "sha256": item_digest(),
                "canonical_uri": (
                    f"gs://project-a-artifacts/candidate-generations/{GENERATION}/items.json"
                ),
                "canonical_object_generation": ITEM_GENERATION,
            },
        },
        "catalog": {
            "distance_metric": "DOT_PRODUCT_DISTANCE",
            "items": catalog_rows(),
        },
        "queries": [
            {"id": f"private-query-{position:03d}", "embedding": [1.0, 0.0]}
            for position in range(20)
        ],
    }


class FakeNamespace:
    def __init__(self, name, allow_tokens, deny_tokens):
        self.name = name
        self.allow_tokens = allow_tokens
        self.deny_tokens = deny_tokens


class FakeQueryEndpoint:
    def __init__(
        self,
        *,
        failure: Exception | None = None,
        generation: str = GENERATION,
        positions=(99, 98, 97),
    ):
        self.resource_name = ENDPOINT
        self.failure = failure
        self.generation = generation
        self.positions = positions
        self.calls: list[dict] = []

    def find_neighbors(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return [[
            SimpleNamespace(id=f"{self.generation}::item-{position:03d}")
            for position in self.positions
        ]]


def config(artifact) -> BenchmarkConfig:
    return BenchmarkConfig(
        binding=artifact.expected_active.binding,
        active_pointer_table=TABLE,
        generation_id=GENERATION,
        promotion_epoch=7,
        client_region="europe-west3",
        top_k=3,
        warmups=2,
        query_limit=None,
        artifact_sha256="e" * 64,
        repository_commit="f" * 40,
    )


def control_snapshot(document: dict, observed_at: str) -> dict:
    pointer = copy.deepcopy(document["expected_active_pointer"])
    manifest = document["manifest_contract"]
    pointer_sha = hashlib.sha256(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "observed_at": observed_at,
        "active_pointer": pointer,
        "active_pointer_sha256": pointer_sha,
        "promotion_manifest": {
            "identity": copy.deepcopy(pointer["manifest"]),
            "schema_version": 3,
            "generation_id": GENERATION,
            "dimensions": 2,
            "items": {
                **copy.deepcopy(manifest["items"]),
                "catalog_payload_match": True,
            },
            "provenance": copy.deepcopy(pointer["provenance"]),
        },
        "vertex": {
            **copy.deepcopy(pointer["serving"]),
            "dimensions": 2,
            "distance_metric": "DOT_PRODUCT_DISTANCE",
            "index_update_method": "STREAM_UPDATE",
            "deployed_mapping_verified": True,
            "endpoint_region": "europe-west3",
        },
    }


class SnapshotReader:
    def __init__(self, document: dict, *, change_after: bool = False, fail_after=False):
        self.document = document
        self.change_after = change_after
        self.fail_after = fail_after
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.fail_after and self.calls == 2:
            raise PermissionError("postflight denied")
        result = control_snapshot(
            self.document,
            f"2026-08-20T12:00:0{self.calls}Z",
        )
        if self.change_after and self.calls == 2:
            result["vertex"]["dimensions"] = 3
        return result


class FakeBigQueryJob:
    def __init__(self, rows):
        self.rows = rows

    def result(self):
        return self.rows


class FakeBigQuery:
    def __init__(self, row):
        self.row = row
        self.queries: list[str] = []

    def query(self, sql):
        self.queries.append(sql)
        return FakeBigQueryJob([self.row])


def active_row(document: dict) -> dict:
    pointer = document["expected_active_pointer"]
    return {
        "generation_id": pointer["generation_id"],
        "epoch": pointer["epoch"],
        "previous_generation_id": pointer["previous_generation_id"],
        "manifest_uri": pointer["manifest"]["uri"],
        "manifest_object_generation": pointer["manifest"]["object_generation"],
        "manifest_sha256": pointer["manifest"]["sha256"],
        "ops_image_uri": pointer["provenance"]["ops_image_uri"],
        "source_revision": pointer["provenance"]["source_revision"],
        "retained_generation_ids": pointer["retained_generation_ids"],
        "champion_score": pointer["champion_score"],
        "index_name": pointer["serving"]["index_resource"],
        "index_endpoint": pointer["serving"]["index_endpoint"],
        "deployed_index_id": pointer["serving"]["deployed_index_id"],
        "promoted_at": datetime(2026, 8, 20, 11, 30, tzinfo=timezone.utc),
        "lease_epoch": pointer["fence_epoch"],
        "lease_owner": None,
        "lease_phase": None,
        "pending_generation_id": None,
        "pending_manifest_uri": None,
        "pending_manifest_object_generation": None,
        "pending_manifest_sha256": None,
        "pending_ops_image_uri": None,
        "pending_source_revision": None,
        "pending_score": None,
    }


class FakeBlob:
    def __init__(self, payload: bytes, generation: int):
        self.payload = payload
        self.generation = generation
        self.download_generations: list[int] = []
        self.reload_count = 0

    def reload(self):
        self.reload_count += 1

    def download_as_bytes(self, *, if_generation_match):
        self.download_generations.append(if_generation_match)
        return self.payload


class FakeBucket:
    def __init__(self, blob: FakeBlob):
        self.fake_blob = blob
        self.names: list[str] = []

    def blob(self, name):
        self.names.append(name)
        return self.fake_blob


class FakeStorage:
    def __init__(self, blob: FakeBlob):
        self.fake_bucket = FakeBucket(blob)
        self.bucket_names: list[str] = []

    def bucket(self, name):
        self.bucket_names.append(name)
        return self.fake_bucket


class FakeIndexService:
    def __init__(self, document=None):
        self.document = document or {
            "name": INDEX,
            "metadata": {
                "config": {
                    "dimensions": 2,
                    "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
                }
            },
            "indexUpdateMethod": "STREAM_UPDATE",
        }
        self.names: list[str] = []

    def get_index(self, *, name):
        self.names.append(name)
        return self.document


class FakeEndpointService:
    def __init__(self, document=None):
        self.document = document or {
            "name": ENDPOINT,
            "deployedIndexes": [{"id": DEPLOYED, "index": INDEX}],
        }
        self.names: list[str] = []

    def get_index_endpoint(self, *, name):
        self.names.append(name)
        return self.document


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_schemas_are_valid_and_all_typed_objects_are_closed(self):
        directory = Path(__file__).resolve().parents[1]
        for name in ("input.schema.json", "result.schema.json", "diagnostic.schema.json"):
            with self.subTest(name=name):
                schema = json.loads((directory / name).read_text())

                def walk(value):
                    if isinstance(value, dict):
                        if value.get("type") == "object":
                            self.assertIs(value.get("additionalProperties"), False)
                        for nested in value.values():
                            walk(nested)
                    elif isinstance(value, list):
                        for nested in value:
                            walk(nested)

                walk(schema)

    def test_reuses_the_production_generation_namespace(self):
        self.assertEqual(GENERATION_NAMESPACE, PRODUCTION_NAMESPACE)
        self.assertEqual(GENERATION_NAMESPACE, "generation")

    def test_rejects_synthetic_and_unlabeled_provenance(self):
        with self.assertRaisesRegex(BenchmarkContractError, "schema violation"):
            parse_artifact(artifact_document(synthetic=True))
        document = artifact_document()
        document["source"]["provenance_verification"] = "verified"
        with self.assertRaisesRegex(BenchmarkContractError, "schema violation"):
            parse_artifact(document)

    def test_catalog_count_and_canonical_hash_prove_completeness(self):
        artifact = parse_artifact(artifact_document())
        self.assertEqual(len(artifact.catalog), 100)
        self.assertEqual(artifact.catalog_payload_sha256, item_digest())
        changed = artifact_document()
        changed["catalog"]["items"][0]["embedding"] = [999.0, 1.0]
        with self.assertRaisesRegex(BenchmarkContractError, "canonical SHA-256"):
            parse_artifact(changed)
        incomplete = artifact_document()
        incomplete["catalog"]["items"].pop()
        with self.assertRaisesRegex(BenchmarkContractError, "schema violation"):
            parse_artifact(incomplete)

    def test_manifest_is_strict_and_bound_to_catalog(self):
        artifact = parse_artifact(artifact_document())
        evidence = parse_live_promotion_manifest(manifest_payload(), artifact=artifact)
        self.assertTrue(evidence["items"]["catalog_payload_match"])
        changed = manifest_document()
        changed["unexpected"] = True
        with self.assertRaisesRegex(BenchmarkContractError, "keys differ"):
            parse_live_promotion_manifest(
                json.dumps(changed).encode(), artifact=artifact
            )

    def test_exact_top_k_is_deterministic(self):
        artifact = parse_artifact(artifact_document())
        self.assertEqual(
            exact_top_k([1.0, 0.0], artifact.catalog, "DOT_PRODUCT_DISTANCE", 3),
            ["item-099", "item-098", "item-097"],
        )

    def test_pass_report_is_schema_valid_and_omits_query_ids(self):
        document = artifact_document()
        artifact = parse_artifact(document)
        endpoint = FakeQueryEndpoint()
        reader = SnapshotReader(document)
        report = run_benchmark(
            config=config(artifact),
            artifact=artifact,
            endpoint=endpoint,
            namespace_factory=FakeNamespace,
            sdk_version="1.165.1",
            control_reader=reader,
        )
        validate_report(report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(reader.calls, 2)
        self.assertEqual(report["ann_fidelity"]["mean_overlap_at_k"], 1.0)
        serialized = json.dumps(report)
        self.assertNotIn("private-query-", serialized)
        self.assertNotIn('"query_id"', serialized)
        self.assertEqual(report["privacy"]["query_identifiers"], "omitted")
        for call in endpoint.calls:
            self.assertEqual(call["deployed_index_id"], DEPLOYED)
            self.assertEqual(call["num_neighbors"], 3)
            namespace = call["filter"][0]
            self.assertEqual(namespace.name, "generation")
            self.assertEqual(namespace.allow_tokens, [GENERATION])
            self.assertEqual(namespace.deny_tokens, [])

    def test_query_failure_generates_a_valid_failed_fixture_without_fallback(self):
        document = artifact_document()
        artifact = parse_artifact(document)
        report = run_benchmark(
            config=config(artifact),
            artifact=artifact,
            endpoint=FakeQueryEndpoint(failure=PermissionError("denied")),
            namespace_factory=FakeNamespace,
            sdk_version="1.165.1",
            control_reader=SnapshotReader(document),
        )
        validate_report(report)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"]["stage"], "querying")
        self.assertEqual(report["workload"]["queries_success"], 0)
        self.assertIsNone(report["latency_ms"]["p50"])
        self.assertTrue(all(not sample["success"] for sample in report["samples"]))

    def test_postflight_change_or_unreadable_state_fails_closed(self):
        document = artifact_document()
        artifact = parse_artifact(document)
        for reader in (
            SnapshotReader(document, change_after=True),
            SnapshotReader(document, fail_after=True),
        ):
            with self.subTest(reader=type(reader).__name__, fail=reader.fail_after):
                report = run_benchmark(
                    config=config(artifact),
                    artifact=artifact,
                    endpoint=FakeQueryEndpoint(),
                    namespace_factory=FakeNamespace,
                    sdk_version="1.165.1",
                    control_reader=reader,
                )
                validate_report(report)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["failure"]["stage"], "control_postflight")

    def test_report_invariants_reject_tampered_counters(self):
        document = artifact_document()
        artifact = parse_artifact(document)
        report = run_benchmark(
            config=config(artifact),
            artifact=artifact,
            endpoint=FakeQueryEndpoint(),
            namespace_factory=FakeNamespace,
            sdk_version="1.165.1",
            control_reader=SnapshotReader(document),
        )
        report["workload"]["queries_success"] -= 1
        with self.assertRaisesRegex(BenchmarkContractError, "counters"):
            validate_report(report)

    def test_active_pointer_read_rejects_lease_and_matches_exact_identity(self):
        document = artifact_document()
        artifact = parse_artifact(document)
        bq = FakeBigQuery(active_row(document))
        actual = read_active_pointer(bq, artifact.expected_active)
        self.assertEqual(actual, artifact.expected_active)
        self.assertIn(TABLE, bq.queries[0])
        leased = active_row(document)
        leased["lease_owner"] = "other-owner"
        with self.assertRaisesRegex(BenchmarkContractError, "in-progress"):
            read_active_pointer(FakeBigQuery(leased), artifact.expected_active)

    def test_manifest_download_is_generation_pinned_and_digest_checked(self):
        artifact = parse_artifact(artifact_document())
        blob = FakeBlob(manifest_payload(), MANIFEST_GENERATION)
        payload = read_immutable_manifest(FakeStorage(blob), artifact.expected_active.manifest)
        self.assertEqual(payload, manifest_payload())
        self.assertEqual(blob.download_generations, [MANIFEST_GENERATION])
        bad = FakeBlob(b"{}", MANIFEST_GENERATION)
        with self.assertRaisesRegex(BenchmarkContractError, "digest"):
            read_immutable_manifest(FakeStorage(bad), artifact.expected_active.manifest)

    def test_live_vertex_contract_validates_all_serving_properties(self):
        binding = parse_artifact(artifact_document()).expected_active.binding
        evidence = read_live_vertex_contract(
            FakeIndexService(),
            FakeEndpointService(),
            binding=binding,
            dimensions=2,
            distance_metric="DOT_PRODUCT_DISTANCE",
        )
        self.assertEqual(evidence["index_update_method"], "STREAM_UPDATE")
        self.assertTrue(evidence["deployed_mapping_verified"])
        wrong = FakeIndexService()
        wrong.document["indexUpdateMethod"] = "BATCH_UPDATE"
        with self.assertRaisesRegex(BenchmarkContractError, "STREAM_UPDATE"):
            read_live_vertex_contract(
                wrong,
                FakeEndpointService(),
                binding=binding,
                dimensions=2,
                distance_metric="DOT_PRODUCT_DISTANCE",
            )

    def test_capture_control_snapshot_uses_each_control_reader_once(self):
        document = artifact_document()
        artifact = parse_artifact(document)
        bq = FakeBigQuery(active_row(document))
        blob = FakeBlob(manifest_payload(), MANIFEST_GENERATION)
        index_service = FakeIndexService()
        endpoint_service = FakeEndpointService()
        cloud = CloudClients(
            bigquery=bq,
            storage=FakeStorage(blob),
            index_service=index_service,
            endpoint_service=endpoint_service,
            query_endpoint=FakeQueryEndpoint(),
            namespace_factory=FakeNamespace,
            sdk_version="1.165.1",
        )
        snapshot = capture_control_snapshot(
            cloud,
            artifact,
            now=lambda: datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot["active_pointer"], document["expected_active_pointer"])
        self.assertEqual(len(bq.queries), 1)
        self.assertEqual(blob.reload_count, 1)
        self.assertEqual(index_service.names, [INDEX])
        self.assertEqual(endpoint_service.names, [ENDPOINT])

    def test_plan_has_hard_caps_dynamic_exact_phrase_and_no_cloud_dry_run(self):
        document = artifact_document()
        artifact = parse_artifact(document)
        plan = build_plan(config(artifact), artifact)
        self.assertEqual(
            plan["planned_cloud_api_calls"]["total_potentially_billable_calls"], 33
        )
        self.assertEqual(
            plan["required_confirmation_phrase"],
            f"EXECUTE 33 POTENTIALLY BILLED CLOUD API CALLS FOR {GENERATION}@7",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "private.runtime-input.json"
            output_path = root / "plan.json"
            input_path.write_text(json.dumps(document))
            argv = self._cli_args(input_path, output_path)
            with mock.patch(
                "benchmarks.vector_search_runtime.runtime_benchmark.credentialed_cloud",
                side_effect=AssertionError("dry run called cloud"),
            ):
                self.assertEqual(main(argv), 0)
            diagnostic = json.loads(output_path.read_text())
            self.assertEqual(diagnostic["status"], "planned")
            self.assertEqual(diagnostic["plan"], plan)

    def test_wrong_live_confirmation_writes_atomic_refusal_without_cloud(self):
        document = artifact_document()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "private.runtime-input.json"
            output_path = root / "report.json"
            input_path.write_text(json.dumps(document))
            argv = self._cli_args(input_path, output_path) + [
                "--execute-live",
                "--attest-source-is-production",
                "--confirmation",
                "wrong phrase",
            ]
            with mock.patch(
                "benchmarks.vector_search_runtime.runtime_benchmark.credentialed_cloud",
                side_effect=AssertionError("wrong confirmation called cloud"),
            ), mock.patch(
                "benchmarks.vector_search_runtime.runtime_benchmark.repository_commit_and_clean",
                side_effect=AssertionError("wrong confirmation inspected repo"),
            ):
                self.assertEqual(main(argv), 2)
            diagnostic = json.loads(output_path.read_text())
            self.assertEqual(diagnostic["status"], "refused")
            self.assertEqual(diagnostic["stage"], "authorization")
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            self.assertEqual(list(root.glob(".report.json.*.tmp")), [])

    def test_atomic_writer_replaces_complete_json_with_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("old")
            write_json_atomic(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertFalse(any(Path(directory).glob("*.tmp")))

    @staticmethod
    def _cli_args(input_path: Path, output_path: Path) -> list[str]:
        return [
            "--artifact",
            str(input_path),
            "--output",
            str(output_path),
            "--active-pointer-table",
            TABLE,
            "--index-resource",
            INDEX,
            "--index-endpoint",
            ENDPOINT,
            "--deployed-index-id",
            DEPLOYED,
            "--generation-id",
            GENERATION,
            "--promotion-epoch",
            "7",
            "--client-region",
            "europe-west3",
            "--top-k",
            "3",
            "--warmups",
            "2",
        ]


if __name__ == "__main__":
    unittest.main()

import inspect
import json
import os
import pathlib
import sys
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "training"),
    str(ROOT / "featurestore"),
    str(ROOT / "pipelines"),
]
os.environ.setdefault("PROJECT", "test-project")

import refresh_index  # noqa: E402
import evaluate  # noqa: E402
from gc_generation import (  # noqa: E402
    RetirementRecord,
    fence_gc_target,
    validate_gc_target,
)
from generation_contract import (  # noqa: E402
    ManifestReceipt,
    ServingBinding,
    active_state_receipt_sha256,
    build_manifest,
    canonical_item_payload,
    canonical_user_payload,
    item_blob_name,
    manifest_blob_name,
    manifest_uri,
    namespaced_datapoint_id,
    promotion_decision,
    sha256_bytes,
    user_blob_name,
    validate_generation_id,
    validate_manifest,
)
from promote import (  # noqa: E402
    ActiveState,
    LeaseFence,
    acquire_initialization_lock,
    acquire_lease,
    finalize_promotion,
    initialize_generation,
    read_active_state,
)


BINDING = ServingBinding(
    "projects/test-project/locations/europe-west3/indexes/123",
    "projects/test-project/locations/europe-west3/indexEndpoints/456",
    "items_v1",
)
OPS_IMAGE = (
    "europe-west3-docker.pkg.dev/test-project/reco/ops@sha256:" + "a" * 64
)
SOURCE_REVISION = "b" * 40
ACTIVE_MANIFEST_SHA256 = "c" * 64
CANDIDATE_MANIFEST_SHA256 = "d" * 64


def item_rows():
    return [
        {"id": "i-1", "embedding": [1.0, 2.0]},
        {"id": "i-2", "embedding": [3.0, 4.0]},
    ]


def user_rows():
    return [
        {"user_id": "u-1", "embedding": [4.0, 3.0]},
        {"user_id": "u-2", "embedding": [2.0, 1.0]},
    ]


def manifest_for(generation="gen-42"):
    items_payload, items = canonical_item_payload(item_rows(), dimensions=2)
    users_payload, users = canonical_user_payload(user_rows(), dimensions=2)
    return build_manifest(
        generation_id=generation,
        created_at="2026-08-01T00:00:00+00:00",
        trained_at="2026-08-01T00:00:00+00:00",
        dimensions=2,
        items_payload=items_payload,
        item_count=len(items),
        canonical_bucket="test-project-artifacts",
        canonical_item_generation=11,
        regional_bucket="test-project-vectors",
        regional_item_generation=12,
        users_payload=users_payload,
        user_count=len(users),
        user_object_generation=13,
        model_uri=f"gs://test-project-artifacts/models/{generation}",
        project="test-project",
        binding=BINDING,
        ops_image_uri=OPS_IMAGE,
        source_revision=SOURCE_REVISION,
        canary_item_id="i-1",
        canary_user_id="u-1",
    )


def active_state(**overrides):
    values = {
        "generation_id": "active-gen",
        "epoch": 3,
        "previous_generation_id": "previous-gen",
        "manifest_uri": manifest_uri("test-project-artifacts", "active-gen"),
        "manifest_object_generation": 21,
        "manifest_sha256": ACTIVE_MANIFEST_SHA256,
        "ops_image_uri": OPS_IMAGE,
        "source_revision": SOURCE_REVISION,
        "retained_generation_ids": ("previous-gen", "active-gen"),
        "champion_score": 0.5,
        "binding": BINDING,
        "promoted_at": datetime.now(timezone.utc),
        "lease_owner": None,
        "lease_epoch": 5,
        "lease_phase": None,
        "lease_acquired_at": None,
        "lease_heartbeat_at": None,
        "lease_expires_at": None,
        "pending_generation_id": None,
        "pending_manifest_uri": None,
        "pending_manifest_object_generation": None,
        "pending_manifest_sha256": None,
        "pending_ops_image_uri": None,
        "pending_source_revision": None,
        "pending_score": None,
    }
    values.update(overrides)
    return ActiveState(**values)


def retirement_record(generation_id="retired-gen", **overrides):
    now = datetime.now(timezone.utc)
    values = {
        "generation_id": generation_id,
        "retired_at": now - timedelta(days=2),
        "drain_until": now - timedelta(days=1),
        "retired_epoch": 2,
        "successor_generation_id": "active-gen",
        "receipt": ManifestReceipt(
            manifest_uri("test-project-artifacts", generation_id),
            19,
            CANDIDATE_MANIFEST_SHA256,
        ),
        "binding": BINDING,
        "ops_image_uri": OPS_IMAGE,
        "source_revision": SOURCE_REVISION,
    }
    values.update(overrides)
    return RetirementRecord(**values)


class FakeJob:
    def result(self):
        return []


class FakeBigQuery:
    def __init__(self):
        self.calls = []

    def query(self, sql, job_config=None):
        self.calls.append((sql, job_config))
        return FakeJob()


class GenerationContractTests(unittest.TestCase):
    def test_paths_hashes_and_manifest_are_generation_bound(self):
        generation = "gen-42"
        manifest = manifest_for(generation)
        validated = validate_manifest(
            json.loads(json.dumps(manifest)),
            expected_generation_id=generation,
            canonical_bucket="test-project-artifacts",
            regional_bucket="test-project-vectors",
            expected_binding=BINDING,
            expected_ops_image_uri=OPS_IMAGE,
            expected_source_revision=SOURCE_REVISION,
        )
        self.assertEqual(
            item_blob_name(generation),
            "candidate-generations/gen-42/items.json",
        )
        self.assertEqual(
            user_blob_name(generation),
            "candidate-generations/gen-42/users.json",
        )
        self.assertEqual(
            manifest_blob_name(generation),
            "candidate-generations/gen-42/manifest.json",
        )
        self.assertEqual(validated["serving"]["index_name"], BINDING.index_name)
        self.assertEqual(validated["items"]["canonical_object_generation"], 11)
        self.assertEqual(validated["users"]["object_generation"], 13)
        self.assertEqual(validated["users"]["dimensions"], 2)
        self.assertEqual(
            validated["items"]["canary_datapoint_id"], "gen-42::i-1"
        )

    def test_manifest_rejects_wrong_exact_serving_target(self):
        wrong = ServingBinding(
            BINDING.index_name,
            BINDING.index_endpoint,
            "another_deployment",
        )
        with self.assertRaisesRegex(ValueError, "different Vertex"):
            validate_manifest(
                manifest_for(),
                expected_generation_id="gen-42",
                canonical_bucket="test-project-artifacts",
                regional_bucket="test-project-vectors",
                expected_binding=wrong,
                expected_ops_image_uri=OPS_IMAGE,
                expected_source_revision=SOURCE_REVISION,
            )

    def test_canonical_user_hash_changes_with_id_vector_or_dimension(self):
        payload, _ = canonical_user_payload(user_rows(), dimensions=2)
        changed, _ = canonical_user_payload(
            [{"user_id": "u-1", "embedding": [9, 3]}, user_rows()[1]],
            dimensions=2,
        )
        self.assertNotEqual(sha256_bytes(payload), sha256_bytes(changed))
        with self.assertRaises(ValueError):
            canonical_user_payload(user_rows(), dimensions=3)

    def test_generation_id_rejects_path_traversal(self):
        for invalid in ("", "../live", "/absolute", "has space", "x" * 65):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_generation_id(invalid)

    def test_gate_is_pure_and_respects_floor_and_margin(self):
        self.assertEqual(
            promotion_decision(score=0.34, floor=0.35, margin=0.03, champion=0.5)[0],
            "hold",
        )
        self.assertEqual(
            promotion_decision(score=0.46, floor=0.35, margin=0.03, champion=0.5)[0],
            "hold",
        )
        self.assertEqual(
            promotion_decision(score=0.47, floor=0.35, margin=0.03, champion=0.5)[0],
            "promote",
        )

    def test_active_receipt_binds_score_manifest_provenance_and_target(self):
        state = active_state()
        common = dict(
            generation_id=state.generation_id,
            epoch=state.epoch,
            manifest_uri_value=state.manifest_uri,
            manifest_object_generation=state.manifest_object_generation,
            manifest_sha256=state.manifest_sha256,
            ops_image_uri=state.ops_image_uri,
            source_revision=state.source_revision,
            champion_score=state.champion_score,
            binding=state.binding,
        )
        receipt = active_state_receipt_sha256(**common)
        self.assertNotEqual(
            receipt,
            active_state_receipt_sha256(**(common | {"champion_score": 0.51})),
        )
        self.assertNotEqual(
            receipt,
            active_state_receipt_sha256(
                **(common | {"manifest_sha256": "e" * 64})
            ),
        )

    def test_legacy_active_row_without_receipt_fails_closed(self):
        row = {
            "generation_id": "active-gen",
            "epoch": 3,
            "manifest_uri": manifest_uri(
                "test-project-artifacts", "active-gen"
            ),
            "manifest_object_generation": None,
            "manifest_sha256": None,
            "ops_image_uri": None,
            "source_revision": None,
        }
        bq = mock.Mock()
        bq.get_table.return_value = object()
        query = mock.Mock()
        query.result.return_value = [row]
        bq.query.return_value = query
        with self.assertRaisesRegex(RuntimeError, "explicit migration"):
            read_active_state(bq)

    def test_evaluation_rejects_legacy_active_schema_before_querying(self):
        bq = mock.Mock()
        bq.get_table.return_value = SimpleNamespace(
            schema=[SimpleNamespace(name="generation_id")]
        )
        with self.assertRaisesRegex(RuntimeError, "explicit receipt migration"):
            evaluate.active_champion(bq, mock.Mock(), BINDING)
        bq.query.assert_not_called()

    def test_gcs_download_reloads_then_pins_the_observed_object_generation(self):
        events = []

        class Blob:
            generation = None

            def reload(self):
                events.append("reload")
                self.generation = 17

            def download_as_bytes(self, *, if_generation_match):
                events.append(("download", if_generation_match))
                return b"payload"

        payload, generation = evaluate.conditional_download(Blob(), 17)
        self.assertEqual((payload, generation), (b"payload", 17))
        self.assertEqual(events, ["reload", ("download", 17)])

    def test_empty_active_table_is_a_safe_first_generation(self):
        bq = mock.Mock()
        bq.get_table.return_value = SimpleNamespace()
        query = mock.Mock()
        query.result.return_value = []
        bq.query.return_value = query
        snapshot = evaluate.active_champion(bq, mock.Mock(), BINDING)
        self.assertEqual(snapshot["generation_id"], "__uninitialized__")
        self.assertEqual(snapshot["epoch"], 0)
        self.assertIsNone(snapshot["champion_score"])
        self.assertEqual(snapshot["manifest_object_generation"], 0)

    def test_active_evaluation_reloads_the_exact_manifest_receipt(self):
        manifest_payload = json.dumps(manifest_for("active-gen")).encode()
        manifest_digest = sha256_bytes(manifest_payload)
        row = {
            "generation_id": "active-gen",
            "epoch": 3,
            "champion_score": 0.5,
            "manifest_uri": manifest_uri(
                "test-project-artifacts", "active-gen"
            ),
            "manifest_object_generation": 21,
            "manifest_sha256": manifest_digest,
            "ops_image_uri": OPS_IMAGE,
            "source_revision": SOURCE_REVISION,
            "index_name": BINDING.index_name,
            "index_endpoint": BINDING.index_endpoint,
            "deployed_index_id": BINDING.deployed_index_id,
        }
        bq = mock.Mock()
        bq.get_table.return_value = object()
        query = mock.Mock()
        query.result.return_value = [row]
        bq.query.return_value = query

        class Blob:
            generation = 21

            def reload(self):
                self.generation = 21

            def download_as_bytes(self, *, if_generation_match):
                self.asserted_generation = if_generation_match
                return manifest_payload

        blob = Blob()
        gcs = mock.Mock()
        gcs.bucket.return_value.blob.return_value = blob
        snapshot = evaluate.active_champion(bq, gcs, BINDING)
        self.assertEqual(snapshot["manifest_sha256"], manifest_digest)
        self.assertEqual(blob.asserted_generation, 21)


class GenerationFilteredIndexTests(unittest.TestCase):
    def test_upsert_namespaces_ids_and_adds_generation_restrict_without_delete(self):
        index = mock.Mock()
        fence = mock.Mock()
        with mock.patch.object(
            refresh_index, "_vertex_datapoints", side_effect=lambda specs: specs
        ):
            specs = refresh_index.upsert_generation(
                index,
                rows=item_rows(),
                generation_id="candidate-gen",
                fence=fence,
                batch_size=1,
            )
        self.assertEqual(
            [spec["datapoint_id"] for spec in specs],
            ["candidate-gen::i-1", "candidate-gen::i-2"],
        )
        self.assertEqual(
            specs[0]["restricts"],
            [{"namespace": "generation", "allow_list": ["candidate-gen"]}],
        )
        self.assertEqual(index.upsert_datapoints.call_count, 2)
        self.assertFalse(index.remove_datapoints.called)
        self.assertEqual(fence.call_count, 2)

    def test_visibility_reads_every_vector_restrict_and_manifest_canary(self):
        specs = refresh_index.datapoint_specs(item_rows(), "candidate-gen")

        class Endpoint:
            def __init__(self):
                self.read_ids = []
                self.query = None

            def read_index_datapoints(self, *, deployed_index_id, ids):
                self.read_ids.extend(ids)
                return [
                    SimpleNamespace(
                        datapoint_id=spec["datapoint_id"],
                        feature_vector=spec["feature_vector"],
                        restricts=[
                            SimpleNamespace(
                                namespace="generation", allow_list=["candidate-gen"]
                            )
                        ],
                    )
                    for spec in specs
                    if spec["datapoint_id"] in ids
                ]

            def find_neighbors(self, **kwargs):
                self.query = kwargs
                return [[SimpleNamespace(id=spec["datapoint_id"]) for spec in specs]]

        endpoint = Endpoint()
        fence = mock.Mock()
        refresh_index.wait_for_generation_visibility(
            endpoint,
            deployed_index_id="items_v1",
            specs=specs,
            generation_id="candidate-gen",
            canary_datapoint_id="candidate-gen::i-1",
            fence=fence,
            batch_size=1,
        )
        self.assertEqual(set(endpoint.read_ids), {spec["datapoint_id"] for spec in specs})
        self.assertEqual(endpoint.query["deployed_index_id"], "items_v1")
        namespace = endpoint.query["filter"][0]
        self.assertEqual(namespace.name, "generation")
        self.assertEqual(namespace.allow_tokens, ["candidate-gen"])
        self.assertGreaterEqual(fence.call_count, 5)

    def test_partial_visibility_times_out_without_deleting_prior_generation(self):
        specs = refresh_index.datapoint_specs(item_rows(), "candidate-gen")
        endpoint = mock.Mock()
        endpoint.read_index_datapoints.return_value = []
        with self.assertRaises(TimeoutError):
            refresh_index.wait_for_generation_visibility(
                endpoint,
                deployed_index_id="items_v1",
                specs=specs,
                generation_id="candidate-gen",
                canary_datapoint_id="candidate-gen::i-1",
                fence=lambda: None,
                timeout_seconds=0,
                poll_seconds=0,
            )
        self.assertFalse(endpoint.remove_datapoints.called)
        self.assertFalse(endpoint.find_neighbors.called)

    def test_dot_product_canary_does_not_require_self_as_nearest_neighbor(self):
        specs = refresh_index.datapoint_specs(
            [
                {"id": "canary", "embedding": [0.1, 0.0]},
                {"id": "large-norm", "embedding": [100.0, 0.0]},
            ],
            "candidate-gen",
        )
        endpoint = mock.Mock()
        endpoint.find_neighbors.return_value = [
            [SimpleNamespace(id="candidate-gen::large-norm")]
        ]
        refresh_index.verify_restricted_canary(
            endpoint,
            deployed_index_id="items_v1",
            specs=specs,
            generation_id="candidate-gen",
            canary_datapoint_id="candidate-gen::canary",
            fence=lambda: None,
        )

    def test_live_vertex_target_checks_mapping_dimension_distance_and_update_mode(self):
        index = SimpleNamespace(
            resource_name=BINDING.index_name,
            sync=mock.Mock(),
            gca_resource=SimpleNamespace(
                name=BINDING.index_name,
                index_update_method="STREAM_UPDATE",
                metadata={
                    "config": {
                        "dimensions": 2,
                        "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
                    }
                },
            ),
        )
        endpoint = SimpleNamespace(
            resource_name=BINDING.index_endpoint,
            sync=mock.Mock(),
            gca_resource=SimpleNamespace(
                name=BINDING.index_endpoint,
                deployed_indexes=[
                    SimpleNamespace(id="items_v1", index=BINDING.index_name)
                ],
            ),
        )
        refresh_index.verify_live_vertex_target(
            index,
            endpoint,
            expected_index_name=BINDING.index_name,
            expected_endpoint_name=BINDING.index_endpoint,
            deployed_index_id="items_v1",
            expected_dimensions=2,
            fence=lambda: None,
        )
        endpoint.gca_resource.deployed_indexes[0].index = (
            "projects/test-project/locations/europe-west3/indexes/wrong"
        )
        with self.assertRaisesRegex(RuntimeError, "maps to another"):
            refresh_index.verify_live_vertex_target(
                index,
                endpoint,
                expected_index_name=BINDING.index_name,
                expected_endpoint_name=BINDING.index_endpoint,
                deployed_index_id="items_v1",
                expected_dimensions=2,
                fence=lambda: None,
            )
        endpoint.gca_resource.deployed_indexes[0].index = BINDING.index_name
        index.gca_resource.metadata["config"]["distanceMeasureType"] = (
            "COSINE_DISTANCE"
        )
        with self.assertRaisesRegex(RuntimeError, "DOT_PRODUCT"):
            refresh_index.verify_live_vertex_target(
                index,
                endpoint,
                expected_index_name=BINDING.index_name,
                expected_endpoint_name=BINDING.index_endpoint,
                deployed_index_id="items_v1",
                expected_dimensions=2,
                fence=lambda: None,
            )


class LeaseAndBootstrapTests(unittest.TestCase):
    def test_acquire_uses_owner_monotonic_epoch_ttl_and_exact_binding(self):
        state = active_state()
        acquired = active_state(
            lease_owner="owner-1",
            lease_epoch=6,
            lease_phase="acquired",
            lease_acquired_at=datetime.now(timezone.utc),
            lease_heartbeat_at=datetime.now(timezone.utc),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            pending_generation_id="candidate-gen",
            pending_manifest_uri=manifest_uri(
                "test-project-artifacts", "candidate-gen"
            ),
            pending_manifest_object_generation=31,
            pending_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
            pending_ops_image_uri=OPS_IMAGE,
            pending_source_revision=SOURCE_REVISION,
            pending_score=0.51,
        )
        bq = FakeBigQuery()
        with mock.patch("promote.read_active_state", side_effect=[state, acquired]):
            fence = acquire_lease(
                bq,
                state=state,
                owner="owner-1",
                candidate_generation="candidate-gen",
                candidate_receipt=ManifestReceipt(
                    acquired.pending_manifest_uri,
                    acquired.pending_manifest_object_generation,
                    acquired.pending_manifest_sha256,
                ),
                candidate_ops_image_uri=OPS_IMAGE,
                candidate_source_revision=SOURCE_REVISION,
                candidate_score=0.51,
                evaluated_active_receipt_sha256=active_state_receipt_sha256(
                    generation_id=state.generation_id,
                    epoch=state.epoch,
                    manifest_uri_value=state.manifest_uri,
                    manifest_object_generation=state.manifest_object_generation,
                    manifest_sha256=state.manifest_sha256,
                    ops_image_uri=state.ops_image_uri,
                    source_revision=state.source_revision,
                    champion_score=state.champion_score,
                    binding=state.binding,
                ),
            )
        self.assertEqual(fence.lease_epoch, 6)
        sql = bq.calls[0][0]
        self.assertIn("COALESCE(lease_epoch, epoch) = @prior_fence_epoch", sql)
        self.assertIn("lease_owner = @owner", sql)
        self.assertIn("lease_phase = 'acquired'", sql)
        self.assertIn("lease_expires_at", sql)
        self.assertIn("index_endpoint = @index_endpoint", sql)

    def test_non_stale_concurrent_lease_fails_closed(self):
        state = active_state(
            lease_owner="other-owner",
            lease_epoch=6,
            lease_phase="propagating",
            lease_acquired_at=datetime.now(timezone.utc),
            lease_heartbeat_at=datetime.now(timezone.utc),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            pending_generation_id="other-candidate",
            pending_manifest_uri=manifest_uri(
                "test-project-artifacts", "other-candidate"
            ),
            pending_manifest_object_generation=30,
            pending_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
            pending_ops_image_uri=OPS_IMAGE,
            pending_source_revision=SOURCE_REVISION,
            pending_score=0.49,
        )
        bq = FakeBigQuery()
        with mock.patch("promote.read_active_state", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "non-stale lease"):
                acquire_lease(
                    bq,
                    state=state,
                    owner="owner-1",
                    candidate_generation="candidate-gen",
                    candidate_receipt=ManifestReceipt(
                        manifest_uri("test-project-artifacts", "candidate-gen"),
                        31,
                        CANDIDATE_MANIFEST_SHA256,
                    ),
                    candidate_ops_image_uri=OPS_IMAGE,
                    candidate_source_revision=SOURCE_REVISION,
                    candidate_score=0.51,
                    evaluated_active_receipt_sha256=active_state_receipt_sha256(
                        generation_id=state.generation_id,
                        epoch=state.epoch,
                        manifest_uri_value=state.manifest_uri,
                        manifest_object_generation=state.manifest_object_generation,
                        manifest_sha256=state.manifest_sha256,
                        ops_image_uri=state.ops_image_uri,
                        source_revision=state.source_revision,
                        champion_score=state.champion_score,
                        binding=state.binding,
                    ),
                )
        self.assertEqual(bq.calls, [])

    def test_stale_recovery_rechecks_the_evaluated_active_receipt(self):
        stale = active_state(
            lease_owner="old-owner",
            lease_epoch=6,
            lease_phase="propagating",
            lease_acquired_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            lease_heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            pending_generation_id="old-candidate",
            pending_manifest_uri=manifest_uri(
                "test-project-artifacts", "old-candidate"
            ),
            pending_manifest_object_generation=30,
            pending_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
            pending_ops_image_uri=OPS_IMAGE,
            pending_source_revision=SOURCE_REVISION,
            pending_score=0.49,
        )
        changed = active_state(manifest_sha256="e" * 64)
        evaluated = active_state_receipt_sha256(
            generation_id=stale.generation_id,
            epoch=stale.epoch,
            manifest_uri_value=stale.manifest_uri,
            manifest_object_generation=stale.manifest_object_generation,
            manifest_sha256=stale.manifest_sha256,
            ops_image_uri=stale.ops_image_uri,
            source_revision=stale.source_revision,
            champion_score=stale.champion_score,
            binding=stale.binding,
        )
        with (
            mock.patch("promote.read_active_state", side_effect=[stale, changed]),
            mock.patch("promote.recover_stale_lease"),
            self.assertRaisesRegex(RuntimeError, "changed during stale"),
        ):
            acquire_lease(
                FakeBigQuery(),
                state=stale,
                owner="new-owner",
                candidate_generation="candidate-gen",
                candidate_receipt=ManifestReceipt(
                    manifest_uri("test-project-artifacts", "candidate-gen"),
                    31,
                    CANDIDATE_MANIFEST_SHA256,
                ),
                candidate_ops_image_uri=OPS_IMAGE,
                candidate_source_revision=SOURCE_REVISION,
                candidate_score=0.51,
                evaluated_active_receipt_sha256=evaluated,
            )

    def test_finalize_switches_pointer_users_and_champion_in_one_transaction(self):
        bq = FakeBigQuery()
        fence = LeaseFence(
            owner="owner-1",
            lease_epoch=6,
            active_generation="active-gen",
            active_epoch=3,
            active_manifest_uri=manifest_uri(
                "test-project-artifacts", "active-gen"
            ),
            active_manifest_object_generation=21,
            active_manifest_sha256=ACTIVE_MANIFEST_SHA256,
            active_ops_image_uri=OPS_IMAGE,
            active_source_revision=SOURCE_REVISION,
            active_champion_score=0.5,
            candidate_generation="candidate-gen",
            candidate_manifest_uri=manifest_uri(
                "test-project-artifacts", "candidate-gen"
            ),
            candidate_manifest_object_generation=31,
            candidate_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
            candidate_ops_image_uri=OPS_IMAGE,
            candidate_source_revision=SOURCE_REVISION,
            candidate_score=0.51,
            binding=BINDING,
        )
        finalize_promotion(bq, fence)
        sql = bq.calls[0][0]
        self.assertIn("BEGIN TRANSACTION", sql)
        self.assertIn("INSERT INTO `test-project.reco.user_embeddings`", sql)
        self.assertIn("generation_id = @candidate_generation", sql)
        self.assertIn("champion_score = @candidate_score", sql)
        self.assertIn("manifest_object_generation = @candidate_manifest_object_generation", sql)
        self.assertIn("pending_manifest_sha256 = @candidate_manifest_sha256", sql)
        self.assertNotIn("lease_epoch = NULL", sql)
        self.assertIn("COMMIT TRANSACTION", sql)

    def test_bootstrap_uses_external_lock_and_idempotent_merge_not_assert_uniqueness(self):
        source = inspect.getsource(initialize_generation)
        self.assertIn("acquire_initialization_lock", source)
        self.assertIn("MERGE", source)
        self.assertNotIn("active generation table must be empty", source)
        self.assertIn('heartbeat("committing", INITIALIZATION_COMMIT_TTL_SECONDS)', source)
        self.assertIn("_assert_initialization_fence", source)
        self.assertIn("manifest_object_generation", source)

    def test_expired_committing_bootstrap_lock_is_never_stolen(self):
        payload = {
            "schema_version": 2,
            "owner": "old-owner",
            "lease_epoch": 4,
            "phase": "committing",
            "generation_id": "old-generation",
            "manifest_uri": manifest_uri(
                "test-project-artifacts", "old-generation"
            ),
            "manifest_object_generation": 41,
            "manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "ops_image_uri": OPS_IMAGE,
            "source_revision": SOURCE_REVISION,
            "score": 0.5,
            "index_name": BINDING.index_name,
            "index_endpoint": BINDING.index_endpoint,
            "deployed_index_id": BINDING.deployed_index_id,
            "acquired_at": (
                datetime.now(timezone.utc) - timedelta(hours=2)
            ).isoformat(),
            "heartbeat_at": (
                datetime.now(timezone.utc) - timedelta(hours=2)
            ).isoformat(),
            "expires_at": (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat(),
        }

        class Blob:
            generation = 9

            def upload_from_string(self, *_args, **_kwargs):
                from google.api_core.exceptions import PreconditionFailed

                raise PreconditionFailed("already exists")

            def reload(self):
                self.generation = 9

            def download_as_bytes(self, *, if_generation_match):
                self.asserted_generation = if_generation_match
                return json.dumps(payload).encode()

        blob = Blob()
        gcs = mock.Mock()
        gcs.bucket.return_value.blob.return_value = blob
        with (
            mock.patch("promote.read_active_state", return_value=None),
            self.assertRaisesRegex(RuntimeError, "non-stealable"),
        ):
            acquire_initialization_lock(
                mock.Mock(),
                gcs,
                owner="new-owner",
                generation_id="candidate-gen",
                binding=BINDING,
                receipt=ManifestReceipt(
                    manifest_uri("test-project-artifacts", "candidate-gen"),
                    31,
                    CANDIDATE_MANIFEST_SHA256,
                ),
                ops_image_uri=OPS_IMAGE,
                source_revision=SOURCE_REVISION,
                score=0.51,
            )


class GarbageCollectionTests(unittest.TestCase):
    def test_gc_refuses_active_previous_or_concurrent_operation(self):
        state = active_state()
        common = dict(
            state=state,
            expected_active_generation="active-gen",
            expected_active_epoch=3,
            binding=BINDING,
        )
        for target in ("active-gen", "previous-gen"):
            with self.subTest(target=target), self.assertRaises(RuntimeError):
                validate_gc_target(
                    generation_id=target,
                    retirement=retirement_record(target),
                    **common,
                )
        busy = active_state(
            lease_owner="owner",
            lease_epoch=6,
            lease_phase="upserting",
            lease_acquired_at=datetime.now(timezone.utc),
            lease_heartbeat_at=datetime.now(timezone.utc),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
            pending_generation_id="candidate-gen",
            pending_manifest_uri=manifest_uri(
                "test-project-artifacts", "candidate-gen"
            ),
            pending_manifest_object_generation=31,
            pending_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
            pending_ops_image_uri=OPS_IMAGE,
            pending_source_revision=SOURCE_REVISION,
            pending_score=0.51,
        )
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            validate_gc_target(
                state=busy,
                retirement=retirement_record(),
                generation_id="retired-gen",
                expected_active_generation="active-gen",
                expected_active_epoch=3,
                binding=BINDING,
            )

    def test_gc_allows_only_an_explicit_retired_generation_fence(self):
        validate_gc_target(
            state=active_state(),
            retirement=retirement_record(),
            generation_id="retired-gen",
            expected_active_generation="active-gen",
            expected_active_epoch=3,
            binding=BINDING,
        )
        self.assertEqual(
            namespaced_datapoint_id("retired-gen", "i-1"), "retired-gen::i-1"
        )

    def test_gc_refuses_retained_or_not_yet_drained_generation(self):
        retained = active_state(
            retained_generation_ids=("retired-gen", "previous-gen", "active-gen")
        )
        with self.assertRaisesRegex(RuntimeError, "retained generation"):
            validate_gc_target(
                state=retained,
                retirement=retirement_record(),
                generation_id="retired-gen",
                expected_active_generation="active-gen",
                expected_active_epoch=3,
                binding=BINDING,
            )
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(RuntimeError, "drain window"):
            validate_gc_target(
                state=active_state(),
                retirement=retirement_record(
                    retired_at=now - timedelta(minutes=1),
                    drain_until=now + timedelta(minutes=5),
                ),
                generation_id="retired-gen",
                expected_active_generation="active-gen",
                expected_active_epoch=3,
                binding=BINDING,
                now=now,
            )

    def test_gc_mutation_fence_binds_retirement_receipt_drain_and_retained_set(self):
        fence = LeaseFence(
            owner="gc-owner",
            lease_epoch=6,
            active_generation="active-gen",
            active_epoch=3,
            active_manifest_uri=manifest_uri(
                "test-project-artifacts", "active-gen"
            ),
            active_manifest_object_generation=21,
            active_manifest_sha256=ACTIVE_MANIFEST_SHA256,
            active_ops_image_uri=OPS_IMAGE,
            active_source_revision=SOURCE_REVISION,
            active_champion_score=0.5,
            candidate_generation="retired-gen",
            candidate_manifest_uri=manifest_uri(
                "test-project-artifacts", "retired-gen"
            ),
            candidate_manifest_object_generation=19,
            candidate_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
            candidate_ops_image_uri=OPS_IMAGE,
            candidate_source_revision=SOURCE_REVISION,
            candidate_score=0.5,
            binding=BINDING,
        )
        bq = FakeBigQuery()
        fence_gc_target(bq, fence, retirement_record())
        sql = bq.calls[0][0]
        self.assertIn("NOT @target_generation IN UNNEST", sql)
        self.assertIn("drain_until <= CURRENT_TIMESTAMP()", sql)
        self.assertIn("manifest_object_generation = @target_manifest_generation", sql)
        self.assertIn("pending_source_revision = @target_source_revision", sql)


class CompiledPipelineTests(unittest.TestCase):
    def test_checked_in_spec_carries_exact_binding_and_evaluation_fence(self):
        spec = json.loads((ROOT / "pipelines" / "reco_retrain.json").read_text())
        self.assertEqual(
            set(spec["root"]["inputDefinitions"]["parameters"]),
            {"generation_id", "index_name", "index_endpoint", "deployed_index_id"},
        )
        executors = spec["deploymentSpec"]["executors"]
        promote_args = executors["exec-promote"]["container"]["args"]
        self.assertIn("--evaluated-active-generation", promote_args)
        self.assertIn("--evaluated-active-epoch", promote_args)
        self.assertIn("--deployed-index-id", promote_args)
        evaluate_args = executors["exec-evaluate"]["container"]["args"]
        self.assertIn("--active-generation-out", evaluate_args)
        self.assertIn("--active-epoch-out", evaluate_args)


if __name__ == "__main__":
    unittest.main()

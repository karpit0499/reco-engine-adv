import unittest
from types import SimpleNamespace

from api.contracts import (
    ContractError,
    bearer_auth_status,
    parse_feedback,
    parse_recommend,
    serving_target,
    strip_generation_datapoint,
    validate_active_receipt,
    verify_vertex_target,
)


REQUEST_ID = "123e4567-e89b-42d3-a456-426614174000"
IMPRESSION_ID = "b5fa62d0-29c4-4e18-b3ca-b2e101cddf2a"
EVENT_ID = "e35ea608-05a0-48e6-9b55-f895600cf23c"
TOKEN = "a-secure-test-token-that-is-long-enough"


class BearerAuthTests(unittest.TestCase):
    def test_matching_bearer_token_is_accepted(self):
        self.assertEqual(bearer_auth_status(TOKEN, f"Bearer {TOKEN}"), "ok")

    def test_missing_or_weak_configuration_fails_closed(self):
        self.assertEqual(bearer_auth_status(None, None), "misconfigured")
        self.assertEqual(bearer_auth_status("too-short", "Bearer too-short"), "misconfigured")

    def test_missing_malformed_or_wrong_credentials_are_rejected(self):
        self.assertEqual(bearer_auth_status(TOKEN, None), "unauthorized")
        self.assertEqual(bearer_auth_status(TOKEN, f"Basic {TOKEN}"), "unauthorized")
        self.assertEqual(bearer_auth_status(TOKEN, "Bearer wrong"), "unauthorized")
        self.assertEqual(bearer_auth_status(TOKEN, f"Bearer  {TOKEN}"), "unauthorized")


class RequestContractTests(unittest.TestCase):
    def test_recommend_contract_accepts_only_bounded_exact_payload(self):
        parsed = parse_recommend({"user_id": "user-42", "request_id": REQUEST_ID})
        self.assertEqual(parsed.user_id, "user-42")
        self.assertEqual(parsed.request_id, REQUEST_ID)

        invalid_payloads = [
            {"user_id": "user-42"},
            {"user_id": "user-42", "request_id": REQUEST_ID, "debug": True},
            {"user_id": "x" * 129, "request_id": REQUEST_ID},
            {"user_id": "user 42", "request_id": REQUEST_ID},
            {"user_id": "user-42", "request_id": REQUEST_ID.upper()},
            ["user-42", REQUEST_ID],
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                parse_recommend(payload)

    def test_feedback_requires_linkage_fields_and_separate_event_uuid(self):
        parsed = parse_feedback({
            "user_id": "user-42",
            "impression_id": IMPRESSION_ID,
            "event_id": EVENT_ID,
            "event": "click",
        })
        self.assertEqual(parsed.impression_id, IMPRESSION_ID)
        self.assertEqual(parsed.event_id, EVENT_ID)

        invalid_payloads = [
            {"user_id": "user-42", "event_id": EVENT_ID, "event": "click"},
            {
                "user_id": "user-42",
                "impression_id": IMPRESSION_ID,
                "event_id": EVENT_ID,
                "event": "view",
            },
            {
                "user_id": "user-42",
                "impression_id": "00000000-0000-0000-0000-000000000000",
                "event_id": EVENT_ID,
                "event": "click",
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                parse_feedback(payload)


class GenerationServingContractTests(unittest.TestCase):
    def test_active_pointer_requires_complete_immutable_receipt_and_provenance(self):
        values = {
            "manifest_uri": "gs://test-artifacts/candidate-generations/gen-42/manifest.json",
            "manifest_object_generation": 17,
            "manifest_sha256": "a" * 64,
            "ops_image_uri": "example.test/reco/ops@sha256:" + "b" * 64,
            "source_revision": "c" * 40,
        }
        validate_active_receipt("gen-42", **values)
        for field, invalid in (
            ("manifest_object_generation", None),
            ("manifest_sha256", "not-a-hash"),
            ("ops_image_uri", "example.test/reco/ops:latest"),
            ("source_revision", "main"),
        ):
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_active_receipt("gen-42", **(values | {field: invalid}))

    def test_exact_vertex_target_and_namespaced_id_are_required(self):
        target = serving_target(
            "projects/p/locations/europe-west3/indexes/123",
            "projects/p/locations/europe-west3/indexEndpoints/456",
            "items_v1",
        )
        self.assertEqual(target.deployed_index_id, "items_v1")
        self.assertEqual(
            strip_generation_datapoint("gen-42", "gen-42::item-7"), "item-7"
        )

    def test_mixed_generation_or_ambiguous_resource_fails_closed(self):
        with self.assertRaises(ContractError):
            strip_generation_datapoint("gen-42", "gen-41::item-7")
        with self.assertRaises(ContractError):
            serving_target("items-index", "items-endpoint", "items_v1")
        with self.assertRaises(ContractError):
            serving_target(
                "projects/p/locations/us-central1/indexes/123",
                "projects/p/locations/europe-west3/indexEndpoints/456",
                "items_v1",
            )

    def test_live_vertex_target_requires_exact_mapping_and_index_config(self):
        target = serving_target(
            "projects/p/locations/europe-west3/indexes/123",
            "projects/p/locations/europe-west3/indexEndpoints/456",
            "items_v1",
        )
        index = SimpleNamespace(
            resource_name=target.index_name,
            gca_resource=SimpleNamespace(
                name=target.index_name,
                metadata={
                    "config": {
                        "dimensions": 32,
                        "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
                    }
                },
                index_update_method=SimpleNamespace(name="STREAM_UPDATE"),
            ),
        )
        endpoint = SimpleNamespace(
            resource_name=target.index_endpoint,
            gca_resource=SimpleNamespace(
                name=target.index_endpoint,
                deployed_indexes=[
                    SimpleNamespace(id="items_v1", index=target.index_name)
                ],
            ),
        )
        verify_vertex_target(index, endpoint, target=target, dimensions=32)

        endpoint.gca_resource.deployed_indexes[0].index = (
            "projects/p/locations/europe-west3/indexes/other"
        )
        with self.assertRaisesRegex(ContractError, "not mapped"):
            verify_vertex_target(index, endpoint, target=target, dimensions=32)

    def test_live_vertex_target_rejects_dimension_distance_or_update_drift(self):
        target = serving_target(
            "projects/p/locations/europe-west3/indexes/123",
            "projects/p/locations/europe-west3/indexEndpoints/456",
            "items_v1",
        )

        def resources(*, dimensions=32, distance="DOT_PRODUCT_DISTANCE", update="STREAM_UPDATE"):
            index = SimpleNamespace(
                resource_name=target.index_name,
                gca_resource=SimpleNamespace(
                    name=target.index_name,
                    metadata={
                        "config": {
                            "dimensions": dimensions,
                            "distanceMeasureType": distance,
                        }
                    },
                    index_update_method=SimpleNamespace(name=update),
                ),
            )
            endpoint = SimpleNamespace(
                resource_name=target.index_endpoint,
                gca_resource=SimpleNamespace(
                    name=target.index_endpoint,
                    deployed_indexes=[
                        SimpleNamespace(id="items_v1", index=target.index_name)
                    ],
                ),
            )
            return index, endpoint

        for kwargs, message in (
            ({"dimensions": 64}, "dimension"),
            ({"distance": "COSINE_DISTANCE"}, "DOT_PRODUCT"),
            ({"update": "BATCH_UPDATE"}, "STREAM_UPDATE"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ContractError, message
            ):
                verify_vertex_target(
                    *resources(**kwargs), target=target, dimensions=32
                )


if __name__ == "__main__":
    unittest.main()

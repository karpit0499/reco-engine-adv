import unittest
from unittest import mock

from api import main
from api.contracts import ContractError, ServingVector, parse_feedback, serving_target


REQUEST_ID = "123e4567-e89b-42d3-a456-426614174000"
IMPRESSION_ID = "b5fa62d0-29c4-4e18-b3ca-b2e101cddf2a"
EVENT_ID = "e35ea608-05a0-48e6-9b55-f895600cf23c"
TOKEN = "a-secure-test-token-that-is-long-enough"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.original_token = main.API_BEARER_TOKEN
        main.API_BEARER_TOKEN = TOKEN
        main.app.config.update(TESTING=True)
        self.client = main.app.test_client()

    def tearDown(self):
        main.API_BEARER_TOKEN = self.original_token

    def test_health_is_non_billable_and_public(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_billable_endpoint_requires_bearer_token(self):
        response = self.client.post(
            "/recommend",
            json={"user_id": "user-42", "request_id": REQUEST_ID},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")

    def test_missing_server_token_fails_closed(self):
        main.API_BEARER_TOKEN = None
        response = self.client.post(
            "/recommend",
            json={"user_id": "user-42", "request_id": REQUEST_ID},
        )
        self.assertEqual(response.status_code, 503)

    def test_recommend_rejects_extra_fields(self):
        response = self.client.post(
            "/recommend",
            json={"user_id": "user-42", "request_id": REQUEST_ID, "admin": True},
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 400)

    @mock.patch.object(
        main,
        "parse_recommend",
        side_effect=ContractError("sensitive parser diagnostics"),
    )
    def test_recommend_contract_errors_do_not_expose_exception_text(self, _parse):
        response = self.client.post(
            "/recommend",
            json={"user_id": "user-42", "request_id": REQUEST_ID},
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "invalid recommendation request"})
        self.assertNotIn("sensitive", response.get_data(as_text=True))

    @mock.patch.object(
        main,
        "parse_feedback",
        side_effect=ContractError("sensitive parser diagnostics"),
    )
    def test_feedback_contract_errors_do_not_expose_exception_text(self, _parse):
        response = self.client.post(
            "/feedback",
            json={
                "user_id": "user-42",
                "impression_id": IMPRESSION_ID,
                "event_id": EVENT_ID,
                "event": "click",
            },
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "invalid feedback request"})
        self.assertNotIn("sensitive", response.get_data(as_text=True))

    @mock.patch.object(main, "log_impression")
    @mock.patch.object(main, "popular_items", return_value=["item-1", "item-2"])
    @mock.patch.object(main, "variant_for", return_value="control")
    def test_client_request_uuid_is_the_returned_impression_id(
        self, _variant, _popular, log_impression
    ):
        log_impression.return_value = REQUEST_ID
        response = self.client.post(
            "/recommend",
            json={"user_id": "user-42", "request_id": REQUEST_ID},
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["impression_id"], REQUEST_ID)
        log_impression.assert_called_once_with(
            "user-42", "control", "popular", ["item-1", "item-2"], REQUEST_ID
        )

    @mock.patch.object(main, "log_impression", side_effect=RuntimeError("write failed"))
    @mock.patch.object(main, "popular_items", return_value=["item-1"])
    @mock.patch.object(main, "variant_for", return_value="control")
    def test_recommend_fails_when_impression_logging_fails(
        self, _variant, _popular, _log_impression
    ):
        response = self.client.post(
            "/recommend",
            json={"user_id": "user-42", "request_id": REQUEST_ID},
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("items", response.json)

    @mock.patch.object(main, "log_feedback")
    @mock.patch.object(main, "impression_belongs_to_user", return_value=False)
    def test_feedback_requires_impression_user_link(self, _linked, log_feedback):
        response = self.client.post(
            "/feedback",
            json={
                "user_id": "user-42",
                "impression_id": IMPRESSION_ID,
                "event_id": EVENT_ID,
                "event": "click",
            },
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 409)
        log_feedback.assert_not_called()

    def test_request_body_limit_is_enforced(self):
        response = self.client.post(
            "/recommend",
            data='{"user_id":"' + ("x" * 3_000) + '"}',
            content_type="application/json",
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 413)


class BigQueryWriteTests(unittest.TestCase):
    def test_impression_uses_client_request_uuid_as_insert_id(self):
        client = mock.Mock()
        client.insert_rows_json.return_value = []
        with mock.patch.object(main, "bq_client", return_value=client):
            result = main.log_impression(
                "user-42", "control", "popular", ["item-1"], REQUEST_ID
            )
        self.assertEqual(result, REQUEST_ID)
        self.assertEqual(client.insert_rows_json.call_args.kwargs["row_ids"], [REQUEST_ID])

    def test_feedback_uses_event_uuid_as_insert_id(self):
        client = mock.Mock()
        client.insert_rows_json.return_value = []
        event = parse_feedback({
            "user_id": "user-42",
            "impression_id": IMPRESSION_ID,
            "event_id": EVENT_ID,
            "event": "conversion",
        })
        with mock.patch.object(main, "bq_client", return_value=client):
            main.log_feedback(event)
        self.assertEqual(client.insert_rows_json.call_args.kwargs["row_ids"], [EVENT_ID])


class GenerationFenceTests(unittest.TestCase):
    def setUp(self):
        self.target = serving_target(
            "projects/p/locations/europe-west3/indexes/123",
            "projects/p/locations/europe-west3/indexEndpoints/456",
            "items_v1",
        )
        self.snapshot = ServingVector(
            embedding=(0.1, 0.2),
            generation_id="old-gen",
            epoch=7,
            target=self.target,
        )

    def test_retrieve_keeps_the_request_snapshot_during_pointer_race(self):
        endpoint = mock.Mock()
        endpoint.find_neighbors.return_value = [[
            mock.Mock(id="old-gen::item-1"),
            mock.Mock(id="old-gen::item-2"),
        ]]
        with mock.patch.object(main, "vs", return_value=endpoint), mock.patch.object(
            main, "bq_client", side_effect=AssertionError("must not re-read pointer")
        ):
            items = main.retrieve(self.snapshot, 2)
        self.assertEqual(items, ["item-1", "item-2"])
        kwargs = endpoint.find_neighbors.call_args.kwargs
        self.assertEqual(kwargs["deployed_index_id"], "items_v1")
        namespace = kwargs["filter"][0]
        self.assertEqual(namespace.name, "generation")
        self.assertEqual(namespace.allow_tokens, ["old-gen"])

    def test_retrieve_rejects_any_mixed_generation_result(self):
        endpoint = mock.Mock()
        endpoint.find_neighbors.return_value = [[mock.Mock(id="new-gen::item-1")]]
        with mock.patch.object(main, "vs", return_value=endpoint):
            with self.assertRaises(ContractError):
                main.retrieve(self.snapshot, 1)

    def test_user_vector_reads_generation_embedding_and_binding_in_one_query(self):
        row = {
            "embedding": [0.1, 0.2],
            "generation_id": "old-gen",
            "epoch": 7,
            "matching_rows": 1,
            "active_rows": 1,
            "index_name": self.target.index_name,
            "index_endpoint": self.target.index_endpoint,
            "deployed_index_id": self.target.deployed_index_id,
            "manifest_uri": "gs://test-artifacts/candidate-generations/old-gen/manifest.json",
            "manifest_object_generation": 17,
            "manifest_sha256": "a" * 64,
            "ops_image_uri": "example.test/reco/ops@sha256:" + "b" * 64,
            "source_revision": "c" * 40,
        }
        query_job = mock.Mock()
        query_job.result.return_value = [row]
        client = mock.Mock()
        client.query.return_value = query_job
        with mock.patch.object(main, "bq_client", return_value=client), \
             mock.patch.object(main, "INDEX_RESOURCE_NAME", self.target.index_name), \
             mock.patch.object(main, "INDEX_ENDPOINT", self.target.index_endpoint), \
             mock.patch.object(main, "DEPLOYED_INDEX_ID", self.target.deployed_index_id):
            snapshot = main.user_vector("user-1")
        self.assertEqual(snapshot, self.snapshot)
        sql = client.query.call_args.args[0]
        self.assertIn("u.generation_id = a.generation_id", sql)
        self.assertIn("a.index_endpoint", sql)

    def test_user_vector_fails_closed_if_active_generation_has_duplicate_user_rows(self):
        row = {
            "embedding": [0.1, 0.2],
            "generation_id": "old-gen",
            "epoch": 7,
            "matching_rows": 2,
            "active_rows": 1,
            "index_name": self.target.index_name,
            "index_endpoint": self.target.index_endpoint,
            "deployed_index_id": self.target.deployed_index_id,
            "manifest_uri": "gs://test-artifacts/candidate-generations/old-gen/manifest.json",
            "manifest_object_generation": 17,
            "manifest_sha256": "a" * 64,
            "ops_image_uri": "example.test/reco/ops@sha256:" + "b" * 64,
            "source_revision": "c" * 40,
        }
        query_job = mock.Mock()
        query_job.result.return_value = [row]
        client = mock.Mock()
        client.query.return_value = query_job
        with mock.patch.object(main, "bq_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "singleton"):
                main.user_vector("user-1")


if __name__ == "__main__":
    unittest.main()

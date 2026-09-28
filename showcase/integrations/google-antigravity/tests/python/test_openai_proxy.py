"""The shim between the Go harness and the model endpoint (src/openai_proxy.py)."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.testclient import TestClient

import openai_proxy


class TestSchemaRepair:
    def test_proto_style_type_names_are_lowercased_at_any_depth(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "todos": {"type": "ARRAY", "items": {"type": "OBJECT"}},
                "count": {"type": ["INTEGER", "NULL"]},
            },
        }
        assert openai_proxy._normalize_schema(schema) == {
            "type": "object",
            "properties": {
                "todos": {"type": "array", "items": {"type": "object"}},
                "count": {"type": ["integer", "null"]},
            },
        }

    def test_a_property_named_type_is_not_mistaken_for_a_type_keyword(self):
        # `properties.type` is a field definition, not a JSON Schema type.
        schema = {"properties": {"type": {"type": "STRING"}}}
        assert openai_proxy._normalize_schema(schema) == {
            "properties": {"type": {"type": "string"}}
        }

    def test_unknown_type_names_are_left_alone(self):
        assert openai_proxy._normalize_schema({"type": "Custom"}) == {"type": "Custom"}

    def test_tool_parameters_and_response_format_are_repaired(self):
        body = json.dumps(
            {
                "tools": [
                    {"function": {"name": "f", "parameters": {"type": "OBJECT"}}}
                ],
                "response_format": {"json_schema": {"schema": {"type": "STRING"}}},
            }
        ).encode()
        repaired = json.loads(openai_proxy._repair_body(body))
        assert repaired["tools"][0]["function"]["parameters"] == {"type": "object"}
        assert repaired["response_format"]["json_schema"]["schema"] == {
            "type": "string"
        }

    def test_a_body_with_nothing_to_repair_is_forwarded_byte_for_byte(self):
        body = b'{"model": "m",  "messages": []}'
        assert openai_proxy._repair_body(body) is body

    def test_a_non_json_body_is_forwarded_unchanged(self):
        assert openai_proxy._repair_body(b"not json") == b"not json"


def _client_with_upstream(monkeypatch, handler, **kwargs) -> TestClient:
    """Builds the shim with its upstream httpx client routed to `handler`."""
    real_client = httpx.AsyncClient

    def fake_client(*args, **client_kwargs):
        client_kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **client_kwargs)

    monkeypatch.setattr(openai_proxy.httpx, "AsyncClient", fake_client)
    return TestClient(openai_proxy.build_app(**kwargs))


class TestProxy:
    def test_auth_and_the_static_aimock_header_are_attached(self, monkeypatch):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = request.headers
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        client = _client_with_upstream(
            monkeypatch,
            handler,
            api_key="sk-test",
            upstream="http://upstream.test",
            extra_headers={"X-AIMock-Context": "google-antigravity"},
        )
        response = client.post(
            "/v1/chat/completions",
            json={"tools": [{"function": {"parameters": {"type": "OBJECT"}}}]},
        )
        assert response.status_code == 200
        assert seen["path"] == "/v1/chat/completions"
        assert seen["headers"]["authorization"] == "Bearer sk-test"
        assert seen["headers"]["x-aimock-context"] == "google-antigravity"
        assert seen["body"]["tools"][0]["function"]["parameters"] == {
            "type": "object"
        }

    def test_an_unreachable_upstream_is_a_json_502_not_a_text_500(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = _client_with_upstream(
            monkeypatch, handler, api_key="k", upstream="http://upstream.test"
        )
        response = client.post("/v1/chat/completions", json={})
        # The harness parses the body as a completion error; a bare text/plain
        # 500 would be reported as a schema failure instead of an outage.
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "upstream_error"

    def test_an_upstream_error_status_is_passed_through(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "slow down"}})

        client = _client_with_upstream(
            monkeypatch, handler, api_key="k", upstream="http://upstream.test"
        )
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 429
        assert response.json()["error"]["message"] == "slow down"

    def test_the_health_route_does_not_touch_the_upstream(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("health must not call upstream")

        client = _client_with_upstream(
            monkeypatch, handler, api_key="k", upstream="http://upstream.test"
        )
        assert client.get("/__shim_health").json() == {
            "ok": True,
            "upstream": "http://upstream.test",
        }


def test_start_background_refuses_to_run_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai_proxy.start_background(port=0)

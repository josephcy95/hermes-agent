"""Behavior contracts for the official Hermes management API routes."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import nullcontext
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile


CAPABILITY_KEYS = {
    "profileConfigurationRead",
    "profileConfigurationWrite",
    "assistantDocumentsRead",
    "assistantDocumentsWrite",
    "skillsList",
    "skillsRead",
    "skillsControl",
    "schedulesList",
    "schedulesWrite",
    "schedulesRun",
    "schedulesHistory",
    "diagnosticsRead",
}

MANAGEMENT_ROUTES = {
    ("GET", "/api/management/capabilities"),
    ("GET", "/api/management/configuration"),
    ("POST", "/api/management/configuration/validate"),
    ("PATCH", "/api/management/configuration"),
    ("GET", "/api/management/documents"),
    ("GET", "/api/management/documents/{document_id}"),
    ("PATCH", "/api/management/documents/{document_id}"),
    ("POST", "/api/management/documents/{document_id}/validate"),
    ("POST", "/api/management/documents/{document_id}/revert"),
    ("GET", "/api/management/skills"),
    ("GET", "/api/management/skills/{skill_id}"),
    ("PATCH", "/api/management/skills/{skill_id}"),
    ("GET", "/api/management/schedules"),
    ("POST", "/api/management/schedules"),
    ("POST", "/api/management/schedules/validate"),
    ("GET", "/api/management/schedules/{schedule_id}"),
    ("PATCH", "/api/management/schedules/{schedule_id}"),
    ("DELETE", "/api/management/schedules/{schedule_id}"),
    ("POST", "/api/management/schedules/{schedule_id}/run"),
    ("GET", "/api/management/schedules/{schedule_id}/history"),
    ("GET", "/api/management/diagnostics"),
}


def _adapter(api_key: str = "") -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {})
    )


def _management_app(
    adapter: APIServerAdapter,
    *,
    multiplex: bool = False,
) -> web.Application:
    middlewares = [adapter._make_profile_prefix_middleware()] if multiplex else []
    app = web.Application(middlewares=middlewares)
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        if not path.startswith("/api/management/"):
            continue
        app.router.add_route(method, path, handler)
        if multiplex:
            app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


def test_route_table_exposes_complete_management_contract_and_mirrors():
    adapter = _adapter()
    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}

    assert MANAGEMENT_ROUTES <= routes
    mirrors = {(method, f"/p/{{profile}}{path}") for method, path in routes}
    for method, path in MANAGEMENT_ROUTES:
        assert (method, f"/p/{{profile}}{path}") in mirrors


@pytest.mark.asyncio
async def test_capabilities_are_versioned_individual_booleans():
    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        response = await client.get("/api/management/capabilities")
        payload = await response.json()

    assert response.status == 200
    assert payload["version"] == 1
    assert isinstance(payload["profileId"], str) and payload["profileId"]
    assert set(payload["capabilities"]) == CAPABILITY_KEYS
    assert all(isinstance(value, bool) for value in payload["capabilities"].values())
    assert payload["capabilities"]["diagnosticsRead"] is True
    assert payload["capabilities"]["skillsList"] is True
    assert payload["capabilities"]["skillsRead"] is True
    assert payload["capabilities"]["skillsControl"] is False


@pytest.mark.asyncio
async def test_managed_installation_denies_and_does_not_advertise_config_write(
    monkeypatch,
):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model: original-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED", "1")

    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        capability_response = await client.get("/api/management/capabilities")
        capabilities = await capability_response.json()
        update_response = await client.patch(
            "/api/management/configuration",
            json={"content": "model: changed-model\n", "expectedRevision": "stale"},
        )
        error = await update_response.json()

    assert capability_response.status == 200
    assert capabilities["capabilities"]["profileConfigurationRead"] is True
    assert capabilities["capabilities"]["profileConfigurationWrite"] is False
    assert capabilities["capabilities"]["assistantDocumentsWrite"] is True
    assert update_response.status == 403
    assert error["error"]["code"] == "managementForbidden"
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: original-model\n"


@pytest.mark.asyncio
async def test_management_auth_uses_normalized_error_envelope():
    adapter = _adapter("correct-secret")
    async with TestClient(TestServer(_management_app(adapter))) as client:
        response = await client.get("/api/management/capabilities")
        payload = await response.json()

    assert response.status == 401
    assert payload == {
        "error": {
            "code": "managementForbidden",
            "message": "Authentication is required for Hermes management operations.",
            "retryable": False,
        }
    }


@pytest.mark.asyncio
async def test_configuration_and_document_routes_translate_camel_case(monkeypatch):
    calls: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def fake_service(module_name: str, function_name: str, *args, **kwargs):
        calls.append((module_name, function_name, args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(api_server, "_invoke_management_service", fake_service)
    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        config_response = await client.patch(
            "/api/management/configuration",
            json={"content": {"model": "test-model"}, "expectedRevision": "config-r1"},
        )
        document_response = await client.patch(
            "/api/management/documents/soul",
            json={
                "content": "# Updated\n",
                "expectedRevision": "doc-r1",
                "lineEnding": "lf",
                "finalNewline": True,
            },
        )

    assert config_response.status == 200
    assert document_response.status == 200
    assert calls == [
        (
            "gateway.management_profile",
            "update_profile_configuration",
            ({"model": "test-model"}, "config-r1"),
            {},
        ),
        (
            "gateway.management_profile",
            "update_assistant_document",
            ("soul", "# Updated\n", "doc-r1"),
            {"line_ending": "lf", "final_newline": True},
        ),
    ]


@pytest.mark.asyncio
async def test_profile_configuration_route_preserves_masked_secrets_on_save():
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    secret = "sk-route-test-secret-value-123456789"
    (home / "config.yaml").write_text(
        "model: original-model\n"
        "providers:\n"
        "  custom:\n"
        f"    api_key: {secret}\n",
        encoding="utf-8",
    )

    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        get_response = await client.get("/api/management/configuration")
        loaded = await get_response.json()
        save_response = await client.patch(
            "/api/management/configuration",
            json={
                "content": "model: changed-model\n",
                "expectedRevision": loaded["revision"],
            },
        )
        saved = await save_response.json()

    assert get_response.status == 200
    assert save_response.status == 200
    assert secret not in json.dumps(loaded)
    assert secret not in json.dumps(saved)
    persisted = (home / "config.yaml").read_text(encoding="utf-8")
    assert "model: changed-model" in persisted
    assert secret in persisted
    assert saved["revision"] != loaded["revision"]


@pytest.mark.asyncio
async def test_assistant_document_route_saves_and_detects_stale_revision():
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "SOUL.md").write_text("# Original\n", encoding="utf-8")

    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        get_response = await client.get("/api/management/documents/soul")
        loaded = await get_response.json()
        save_response = await client.patch(
            "/api/management/documents/soul",
            json={
                "content": "# Updated\n",
                "expectedRevision": loaded["revision"],
            },
        )
        saved = await save_response.json()
        conflict_response = await client.patch(
            "/api/management/documents/soul",
            json={
                "content": "# Stale draft\n",
                "expectedRevision": loaded["revision"],
            },
        )
        conflict = await conflict_response.json()

    assert get_response.status == 200
    assert save_response.status == 200
    assert saved["content"] == "# Updated\n"
    assert conflict_response.status == 409
    assert conflict["error"]["code"] == "managementConflict"
    assert conflict["error"]["details"] == {
        "currentRevision": saved["revision"],
        "resource": "assistantDocument",
        "resourceId": "soul",
        "expectedRevision": loaded["revision"],
    }
    assert (home / "SOUL.md").read_text(encoding="utf-8") == "# Updated\n"


@pytest.mark.asyncio
async def test_conflicts_preserve_safe_revision_details_but_not_paths(monkeypatch):
    class RevisionConflict(Exception):
        code = "revision_conflict"
        status_code = 409
        message = "Resource at /private/profile/config.yaml changed; secret=do-not-leak"
        details = {"currentRevision": "config-r2", "localPath": "/private/profile"}

    def conflict(*_args):
        raise RevisionConflict()

    monkeypatch.setattr(api_server, "_invoke_management_service", conflict)
    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        response = await client.patch(
            "/api/management/configuration",
            json={"content": {}, "expectedRevision": "config-r1"},
        )
        payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == "managementConflict"
    assert payload["error"]["message"] == "The resource changed since it was loaded."
    assert payload["error"]["details"] == {
        "currentRevision": "config-r2",
        "resource": "configuration",
        "expectedRevision": "config-r1",
    }
    serialized = json.dumps(payload)
    assert "/private/profile" not in serialized
    assert "do-not-leak" not in serialized


@pytest.mark.asyncio
async def test_read_only_skill_mutation_is_honestly_unsupported():
    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        response = await client.patch(
            "/api/management/skills/example",
            json={"enabled": False, "expectedRevision": "skill-r1"},
        )
        payload = await response.json()

    assert response.status == 501
    assert payload["error"]["code"] == "managementUnsupported"
    assert payload["error"]["retryable"] is False


@pytest.mark.asyncio
async def test_schedule_create_partial_failure_prevents_duplicate_retry(monkeypatch):
    class SchedulerRegistrationFailure(Exception):
        code = "managementUpstreamUnavailable"
        status_code = 424
        message = "External scheduler registration failed."
        details = {
            "scheduleId": "saved-job",
            "jobSaved": True,
            "schedulerRegistered": False,
            "retryCreate": False,
        }

    def fail_registration(*_args, **_kwargs):
        raise SchedulerRegistrationFailure()

    monkeypatch.setattr(api_server, "_invoke_management_service", fail_registration)
    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        response = await client.post(
            "/api/management/schedules",
            json={
                "name": "Saved schedule",
                "prompt": "Run the saved task",
                "schedule": {"kind": "cron", "expression": "0 9 * * *"},
            },
        )
        payload = await response.json()

    assert response.status == 424
    assert payload["error"]["code"] == "managementUpstreamUnavailable"
    assert payload["error"]["message"] == (
        "The schedule was saved, but external scheduler registration failed. "
        "Do not retry creation; refresh schedules."
    )
    assert payload["error"]["retryable"] is False
    assert payload["error"]["details"] == {
        "resource": "schedule",
        "scheduleId": "saved-job",
        "jobSaved": True,
        "schedulerRegistered": False,
        "retryCreate": False,
    }


@pytest.mark.asyncio
async def test_schedule_run_is_accepted_then_executes_in_retained_background_task(
    monkeypatch,
):
    executed = threading.Event()
    calls: list[tuple[str, tuple[object, ...]]] = []
    scoped_homes: list[Path] = []

    def fake_service(_module_name: str, function_name: str, *args):
        from hermes_constants import get_hermes_home

        calls.append((function_name, args))
        scoped_homes.append(get_hermes_home().resolve())
        if function_name == "run_schedule_now":
            return {
                "accepted": True,
                "executionStatus": "queued",
                "schedule": {"id": "daily"},
            }
        if function_name == "execute_schedule_now":
            executed.set()
            return {"executed": True, "id": "daily"}
        raise AssertionError(function_name)

    monkeypatch.setattr(api_server, "_invoke_management_service", fake_service)
    adapter = _adapter()
    default_home = Path(os.environ["HERMES_HOME"])
    profile_home = default_home / "profiles" / "coder"
    profile_home.mkdir(parents=True, exist_ok=True)

    class Runner:
        adapters = {"test": object()}
        config = GatewayConfig(multiplex_profiles=True)

    adapter.gateway_runner = Runner()
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", default_home), ("coder", profile_home)],
    )
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "profile-secret")
    async with TestClient(TestServer(_management_app(adapter, multiplex=True))) as client:
        response = await client.post(
            "/p/coder/api/management/schedules/daily/run",
            json={"expectedRevision": "schedule-r1"},
            headers={"Authorization": "Bearer profile-secret"},
        )
        payload = await response.json()
        assert await asyncio.to_thread(executed.wait, 2)
        await asyncio.sleep(0)

    assert response.status == 202
    assert payload["accepted"] is True
    assert [name for name, _args in calls] == ["run_schedule_now", "execute_schedule_now"]
    assert calls[1][1][0] == "daily"
    assert calls[1][1][1] is Runner.adapters
    assert scoped_homes == [profile_home.resolve(), profile_home.resolve()]
    assert adapter._pending_agent_requests == 0


@pytest.mark.asyncio
async def test_diagnostics_are_redacted_and_omit_runtime_paths(monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "workspace").mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: example/model\n"
        "  provider: example-provider\n"
        "agent:\n"
        "  reasoning_effort: medium\n",
        encoding="utf-8",
    )
    (home / "gateway_state.json").write_text(
        json.dumps(
            {
                "gateway_state": "degraded",
                "argv": [str(home / "bin" / "hermes")],
                "platforms": {
                    "api_server": {
                        "state": "failed",
                        "error_message": "Bearer sk-super-secret-value-123456",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value-123456")

    adapter = _adapter()
    async with TestClient(TestServer(_management_app(adapter))) as client:
        response = await client.get("/api/management/diagnostics")
        payload = await response.json()

    assert response.status == 200
    assert payload["redacted"] is True
    assert payload["workspaceAvailable"] is True
    assert payload["connection"]["state"] == "unhealthy"
    assert payload["activeRuntime"] == {
        "provider": "example-provider",
        "model": "example/model",
        "reasoning": "medium",
    }
    assert set(payload["capabilities"]) == CAPABILITY_KEYS
    serialized = json.dumps(payload)
    assert str(home) not in serialized
    assert "sk-super-secret-value-123456" not in serialized
    assert "error_message" not in serialized
    assert "filesystem paths are omitted" in payload["report"]


@pytest.mark.asyncio
async def test_multiplex_management_route_keeps_profile_context(monkeypatch):
    adapter = _adapter("listener-secret")

    class Runner:
        config = GatewayConfig(multiplex_profiles=True)

    adapter.gateway_runner = Runner()
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", Path("/unused")), ("coder", Path("/unused/coder"))],
    )
    monkeypatch.setattr(adapter, "_profile_scope", lambda _profile: nullcontext())
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "profile-secret")

    def scoped_capabilities():
        return {
            "profileId": _api_request_profile.get(),
            "version": 1,
            "capabilities": dict.fromkeys(CAPABILITY_KEYS, False),
            "checkedAt": "2026-08-11T00:00:00Z",
        }

    monkeypatch.setattr(
        "gateway.management_diagnostics.get_management_capabilities",
        scoped_capabilities,
    )
    async with TestClient(TestServer(_management_app(adapter, multiplex=True))) as client:
        response = await client.get(
            "/p/coder/api/management/capabilities",
            headers={"Authorization": "Bearer profile-secret"},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["profileId"] == "coder"

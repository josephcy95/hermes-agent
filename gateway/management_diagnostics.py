"""Capability discovery and redacted diagnostics for the management API.

This module deliberately returns a small, allowlisted view of the active
profile.  It never serializes configuration, environment variables, runtime
error messages, process arguments, or filesystem paths.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Callable

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home


MANAGEMENT_API_VERSION = 1

CAPABILITY_DEFAULTS: dict[str, bool] = {
    "profileConfigurationRead": False,
    "profileConfigurationWrite": False,
    "assistantDocumentsRead": False,
    "assistantDocumentsWrite": False,
    "skillsList": False,
    "skillsRead": False,
    "skillsControl": False,
    "schedulesList": False,
    "schedulesWrite": False,
    "schedulesRun": False,
    "schedulesHistory": False,
    "diagnosticsRead": True,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any, *, limit: int = 160) -> str:
    """Return a bounded, single-line, aggressively redacted display value."""
    text = redact_sensitive_text(str(value or ""), force=True)
    return " ".join(text.split())[:limit]


def _safe_runtime_selector(value: Any, *, limit: int = 255) -> str:
    """Return a provider/model selector unless it is an absolute/local path."""
    text = _safe_text(value, limit=limit)
    lowered = text.lower()
    if (
        text.startswith(("/", "~/", "\\"))
        or "\\" in text
        or re.match(r"^[A-Za-z]:[/\\]", text)
        or lowered.startswith(("file:", "path:"))
        or "://" in text
    ):
        return ""
    return text


def _profile_id() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return _safe_text(get_active_profile_name(), limit=64) or "default"
    except Exception:
        return "default"


def _module_function(module_name: str, *names: str) -> Callable[..., Any] | None:
    try:
        module = import_module(module_name)
    except Exception:
        return None
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            return value
    return None


def _merge_capability_hint(
    capabilities: dict[str, bool],
    module_name: str,
    function_names: tuple[str, ...],
    aliases: dict[str, str],
) -> None:
    """Merge a service capability hint without trusting unknown keys.

    Service modules use small domain-oriented capability payloads (for
    example ``{"read": True, "control": False}``).  The public protocol is
    intentionally flat, so this function translates only explicitly
    allowlisted aliases and ignores every other value.
    """
    probe = _module_function(module_name, *function_names)
    if probe is None:
        return
    try:
        result = probe()
    except Exception:
        return
    if not isinstance(result, dict):
        return
    for source, target in aliases.items():
        if source in result and target in capabilities:
            capabilities[target] = result[source] is True


def get_management_capabilities() -> dict[str, Any]:
    """Return versioned, per-operation capabilities for the active profile."""
    capabilities = dict(CAPABILITY_DEFAULTS)

    # Function presence is the compatibility floor.  Older installations can
    # omit any one service module and still advertise the remaining sections.
    profile_module = "gateway.management_profile"
    capabilities["profileConfigurationRead"] = _module_function(
        profile_module,
        "get_profile_configuration",
        "read_profile_configuration",
        "get_management_configuration",
    ) is not None
    capabilities["profileConfigurationWrite"] = _module_function(
        profile_module,
        "update_profile_configuration",
        "write_profile_configuration",
        "patch_profile_configuration",
    ) is not None
    capabilities["assistantDocumentsRead"] = (
        _module_function(
            profile_module,
            "list_assistant_documents",
            "list_profile_documents",
            "list_management_documents",
        )
        is not None
        and _module_function(
            profile_module,
            "get_assistant_document",
            "read_assistant_document",
            "get_profile_document",
        )
        is not None
    )
    capabilities["assistantDocumentsWrite"] = _module_function(
        profile_module,
        "update_assistant_document",
        "write_assistant_document",
        "patch_profile_document",
    ) is not None
    _merge_capability_hint(
        capabilities,
        profile_module,
        (
            "get_management_profile_capabilities",
            "get_management_capabilities",
        ),
        {
            "profileConfigurationRead": "profileConfigurationRead",
            "profileConfigurationWrite": "profileConfigurationWrite",
            "assistantDocumentsRead": "assistantDocumentsRead",
            "assistantDocumentsWrite": "assistantDocumentsWrite",
            "configuration_read": "profileConfigurationRead",
            "configuration_write": "profileConfigurationWrite",
            "documents_read": "assistantDocumentsRead",
            "documents_write": "assistantDocumentsWrite",
        },
    )

    skills_module = "gateway.management_skills"
    capabilities["skillsList"] = _module_function(
        skills_module, "list_management_skills"
    ) is not None
    capabilities["skillsRead"] = _module_function(
        skills_module, "read_management_skill"
    ) is not None
    capabilities["skillsControl"] = _module_function(
        skills_module,
        "update_management_skill",
        "control_management_skill",
    ) is not None
    _merge_capability_hint(
        capabilities,
        skills_module,
        ("get_management_skills_capability", "get_management_capabilities"),
        {
            "list": "skillsList",
            "read": "skillsRead",
            "control": "skillsControl",
            "skillsList": "skillsList",
            "skillsRead": "skillsRead",
            "skillsControl": "skillsControl",
        },
    )

    schedules_module = "gateway.management_schedules"
    capabilities["schedulesList"] = _module_function(
        schedules_module, "list_schedules"
    ) is not None
    capabilities["schedulesWrite"] = all(
        _module_function(schedules_module, name) is not None
        for name in ("create_schedule", "update_schedule", "delete_schedule")
    )
    capabilities["schedulesRun"] = _module_function(
        schedules_module, "run_schedule_now"
    ) is not None
    capabilities["schedulesHistory"] = _module_function(
        schedules_module, "list_schedule_history"
    ) is not None
    _merge_capability_hint(
        capabilities,
        schedules_module,
        ("get_management_schedules_capability", "get_management_capabilities"),
        {
            "list": "schedulesList",
            "write": "schedulesWrite",
            "run": "schedulesRun",
            "history": "schedulesHistory",
            "schedulesList": "schedulesList",
            "schedulesWrite": "schedulesWrite",
            "schedulesRun": "schedulesRun",
            "schedulesHistory": "schedulesHistory",
        },
    )

    return {
        "profileId": _profile_id(),
        "version": MANAGEMENT_API_VERSION,
        "capabilities": capabilities,
        "checkedAt": _utc_now(),
    }


def _active_runtime() -> dict[str, str] | None:
    """Return only non-secret runtime selectors from profile configuration."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return None
    if not isinstance(config, dict):
        return None

    model_config = config.get("model")
    provider: Any = None
    model: Any = None
    if isinstance(model_config, str):
        model = model_config
    elif isinstance(model_config, dict):
        model = model_config.get("default") or model_config.get("model")
        provider = model_config.get("provider")

    reasoning: Any = None
    agent_config = config.get("agent")
    if isinstance(agent_config, dict):
        reasoning = agent_config.get("reasoning_effort")

    result: dict[str, str] = {}
    if provider:
        safe_provider = _safe_runtime_selector(provider)
        if safe_provider:
            result["provider"] = safe_provider
    if model:
        safe_model = _safe_runtime_selector(model)
        if safe_model:
            result["model"] = safe_model
    if isinstance(reasoning, (str, int, float)) and str(reasoning).strip():
        result["reasoning"] = _safe_text(reasoning, limit=32)
    return result or None


def _connection_summary(checked_at: str) -> tuple[dict[str, Any], list[str]]:
    """Translate runtime state to a safe connection summary.

    Runtime error messages and process metadata are intentionally ignored.
    """
    warnings: list[str] = []
    try:
        from gateway.status import read_runtime_status

        runtime = read_runtime_status() or {}
    except Exception:
        runtime = {}

    gateway_state = runtime.get("gateway_state") if isinstance(runtime, dict) else None
    state_map = {
        "running": "connected",
        "ready": "connected",
        "starting": "connected",
        "draining": "connected",
        "degraded": "unhealthy",
        "startup_failed": "unhealthy",
        "stopped": "offline",
    }
    state = state_map.get(gateway_state, "connected")
    messages = {
        "connected": "Hermes Agent management API is reachable.",
        "unhealthy": "Hermes Agent is reachable but reports a degraded state.",
        "offline": "Hermes Agent reports that the gateway is offline.",
    }
    if not runtime:
        warnings.append(
            "Gateway runtime status is unavailable; API reachability was used instead."
        )
    return {
        "state": state,
        "message": messages[state],
        "checkedAt": checked_at,
    }, warnings


def _workspace_available() -> bool:
    """Probe the profile workspace without exposing its absolute path."""
    try:
        workspace = get_hermes_home() / "workspace"
        return workspace.is_dir()
    except OSError:
        return False


def _render_report(payload: dict[str, Any]) -> str:
    capabilities = payload["capabilities"]
    runtime = payload.get("activeRuntime") or {}
    lines = [
        "Hermes management diagnostics",
        f"Profile: {payload['profileId']}",
        f"Checked: {payload['checkedAt']}",
        f"Connection: {payload['connection']['state']}",
        f"Agent version: {payload['agentVersion']}",
        f"Management API version: {payload['agentApiVersion']}",
        f"Workspace available: {'yes' if payload['workspaceAvailable'] else 'no'}",
    ]
    if runtime.get("provider"):
        lines.append(f"Provider: {runtime['provider']}")
    if runtime.get("model"):
        lines.append(f"Model: {runtime['model']}")
    if runtime.get("reasoning"):
        lines.append(f"Reasoning: {runtime['reasoning']}")
    lines.append("Capabilities:")
    for name in CAPABILITY_DEFAULTS:
        lines.append(f"- {name}: {'yes' if capabilities.get(name) else 'no'}")
    if payload["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {_safe_text(warning)}" for warning in payload["warnings"])
    lines.append("Sensitive values and filesystem paths are omitted.")
    return "\n".join(lines)


def get_management_diagnostics(agent_version: str = "dev") -> dict[str, Any]:
    """Build a compact diagnostics payload safe to copy or download."""
    capabilities_payload = get_management_capabilities()
    checked_at = capabilities_payload["checkedAt"]
    connection, warnings = _connection_summary(checked_at)
    payload: dict[str, Any] = {
        "profileId": capabilities_payload["profileId"],
        "checkedAt": checked_at,
        "connection": connection,
        "agentVersion": _safe_text(agent_version, limit=64) or "dev",
        "agentApiVersion": str(MANAGEMENT_API_VERSION),
        # The upstream Agent API cannot know the independently-upgradeable
        # Hermes Chat version.  The Chat BFF replaces this sentinel with its
        # own build version before returning the normalized browser payload.
        "chatVersion": "unknown",
        "workspaceAvailable": _workspace_available(),
        "capabilities": capabilities_payload["capabilities"],
        "warnings": warnings,
        "redacted": True,
    }
    runtime = _active_runtime()
    if runtime:
        payload["activeRuntime"] = runtime
    payload["report"] = _render_report(payload)
    return payload


__all__ = [
    "CAPABILITY_DEFAULTS",
    "MANAGEMENT_API_VERSION",
    "get_management_capabilities",
    "get_management_diagnostics",
]

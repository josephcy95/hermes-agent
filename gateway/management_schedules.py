"""Profile-scoped schedule management for authenticated gateway clients.

This module is deliberately a narrow adapter over Hermes' cron model.  It does
not expose script paths, working directories, delivery origins, provider base
URLs, or an arbitrary job dictionary editor.  Prompt-based jobs can be created
and edited; advanced script/monitor jobs remain visible and controllable without
turning the management API into a remote shell configuration surface.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

MAX_NAME_CHARS = 160
MAX_PROMPT_CHARS = 32_000
MAX_SCHEDULE_CHARS = 256
MAX_SKILLS = 32
MAX_SKILL_CHARS = 200
MAX_HISTORY_LIMIT = 100
MAX_ERROR_CHARS = 1_000
MAX_REVISION_CHARS = 256

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_CREATE_FIELDS = frozenset({"name", "prompt", "schedule", "enabled", "skills"})
_UPDATE_FIELDS = frozenset(
    {"name", "prompt", "schedule", "enabled", "skills", "revision", "expectedRevision"}
)


class ScheduleManagementError(Exception):
    """A normalized, client-safe management error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        safe_message = _bounded_text(_redact_text(message), MAX_ERROR_CHARS)
        super().__init__(safe_message)
        self.code = str(code)
        self.message = safe_message
        self.status = int(status)
        # ``status_code`` is a compatibility convenience for HTTP adapters.
        self.status_code = self.status
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        error: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            error["details"] = self.details
        return {"error": error}


def _bounded_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _redact_text(value: Any) -> str:
    text = redact_sensitive_text(
        "" if value is None else str(value),
        force=True,
        redact_url_credentials=True,
    )
    # Stored cron errors from advanced jobs can contain absolute script or
    # workdir paths. Keep profile roots server-side along with credentials.
    try:
        active_home = str(_active_home())
        user_home = str(Path.home().expanduser().resolve())
        if active_home and active_home != "/":
            text = text.replace(active_home, "<HERMES_HOME>")
        if user_home and user_home != "/" and user_home != active_home:
            text = text.replace(user_home, "~")
    except (OSError, RuntimeError):
        pass
    return text


def _active_home() -> Path:
    """Resolve the active profile home; callers cannot select another one."""
    return get_hermes_home().expanduser().resolve()


@contextmanager
def _job_store() -> Iterator[Path]:
    """Pin all cron job operations in this context to the active profile."""
    from cron.jobs import use_cron_store

    home = _active_home()
    with use_cron_store(home):
        yield home


@contextmanager
def _execution_store(home: Optional[Path] = None) -> Iterator[Any]:
    """Pin public execution-ledger calls to one active profile context."""
    from cron import executions
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = (home or _active_home()).resolve()
    token = set_hermes_home_override(profile_home)
    try:
        yield executions
    finally:
        reset_hermes_home_override(token)


def _require_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ScheduleManagementError(
            "managementValidationFailed",
            "Request body must be a JSON object.",
        )
    return payload


def _reject_unknown_fields(payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise ScheduleManagementError(
            "managementValidationFailed",
            f"Unsupported schedule field(s): {', '.join(unknown)}.",
            details={"fields": unknown},
        )


def _required_text(
    payload: Mapping[str, Any], field: str, limit: int, *, allow_empty: bool = False
) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ScheduleManagementError(
            "managementValidationFailed",
            f"'{field}' must be a string.",
            details={"field": field},
        )
    text = value.strip()
    if not text and not allow_empty:
        raise ScheduleManagementError(
            "managementValidationFailed",
            f"'{field}' must not be empty.",
            details={"field": field},
        )
    if len(text) > limit:
        raise ScheduleManagementError(
            "managementResourceTooLarge",
            f"'{field}' exceeds the {limit} character limit.",
            status=413,
            details={"field": field, "maxLength": limit},
        )
    return text


def _optional_text(payload: Mapping[str, Any], field: str, limit: int) -> Optional[str]:
    if field not in payload:
        return None
    return _required_text(payload, field, limit)


def _normalize_skills(value: Any, *, present: bool) -> Optional[List[str]]:
    if not present:
        return None
    if not isinstance(value, list):
        raise ScheduleManagementError(
            "managementValidationFailed",
            "'skills' must be an array of strings.",
            details={"field": "skills"},
        )
    if len(value) > MAX_SKILLS:
        raise ScheduleManagementError(
            "managementResourceTooLarge",
            f"'skills' may contain at most {MAX_SKILLS} entries.",
            status=413,
            details={"field": "skills", "maxItems": MAX_SKILLS},
        )
    normalized: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ScheduleManagementError(
                "managementValidationFailed",
                "Every skill reference must be a string.",
                details={"field": "skills"},
            )
        skill = item.strip()
        if not skill:
            continue
        if len(skill) > MAX_SKILL_CHARS:
            raise ScheduleManagementError(
                "managementResourceTooLarge",
                f"A skill reference exceeds the {MAX_SKILL_CHARS} character limit.",
                status=413,
                details={"field": "skills", "maxLength": MAX_SKILL_CHARS},
            )
        if _redact_text(skill) != skill:
            raise ScheduleManagementError(
                "managementUnsafeResource",
                "A skill reference contains sensitive data.",
                status=422,
                details={"field": "skills"},
            )
        if skill not in normalized:
            normalized.append(skill)
    return normalized


def _optional_enabled(payload: Mapping[str, Any]) -> Optional[bool]:
    if "enabled" not in payload:
        return None
    enabled = payload["enabled"]
    if not isinstance(enabled, bool):
        raise ScheduleManagementError(
            "managementValidationFailed",
            "'enabled' must be a boolean.",
            details={"field": "enabled"},
        )
    return enabled


def _scan_prompt(prompt: str) -> None:
    if _redact_text(prompt) != prompt:
        raise ScheduleManagementError(
            "managementUnsafeResource",
            "The schedule prompt contains sensitive data. Reference a configured "
            "credential by environment variable name instead of storing its value.",
            status=422,
            details={"field": "prompt"},
        )
    # Keep management-created jobs on the same security rail as the cron tool.
    from tools.cronjob_tools import _scan_cron_prompt

    reason = _scan_cron_prompt(prompt)
    if reason:
        raise ScheduleManagementError(
            "managementUnsafeResource",
            reason,
            status=422,
            details={"field": "prompt"},
        )


def _parse_schedule(schedule: str) -> Dict[str, Any]:
    from cron.jobs import compute_next_run, parse_schedule

    try:
        parsed = parse_schedule(schedule)
        next_run_at = compute_next_run(parsed)
    except (TypeError, ValueError) as exc:
        raise ScheduleManagementError(
            "managementValidationFailed",
            str(exc),
            status=422,
            details={"field": "schedule"},
        ) from exc
    if parsed.get("kind") == "once" and next_run_at is None:
        raise ScheduleManagementError(
            "managementValidationFailed",
            "The one-time schedule is too far in the past to run.",
            status=422,
            details={"field": "schedule"},
        )
    return {
        "parsed": parsed,
        "kind": str(parsed.get("kind") or ""),
        "display": str(parsed.get("display") or schedule),
        "nextRunAt": next_run_at,
    }


def _validate_job_id(job_id: Any) -> str:
    text = str(job_id or "").strip()
    if not _JOB_ID_RE.fullmatch(text) or text in {".", ".."}:
        raise ScheduleManagementError(
            "managementValidationFailed",
            "Schedule ID is invalid.",
        )
    return text


def _revision(job: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in job.items() if key != "latest_execution"}
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_revision(payload: Mapping[str, Any]) -> str:
    expected_value = payload.get("expectedRevision")
    alias_value = payload.get("revision")
    if (
        expected_value is not None
        and alias_value is not None
        and expected_value != alias_value
    ):
        raise ScheduleManagementError(
            "managementValidationFailed",
            "Conflicting schedule revision fields were provided.",
            details={"field": "expectedRevision"},
        )
    value = expected_value if expected_value is not None else alias_value
    if not isinstance(value, str) or not value.strip():
        raise ScheduleManagementError(
            "precondition_required",
            "A current schedule revision is required for this change.",
            status=428,
        )
    revision = value.strip()
    if len(revision) > MAX_REVISION_CHARS:
        raise ScheduleManagementError(
            "managementResourceTooLarge",
            "The schedule revision is too large.",
            status=413,
            details={"field": "expectedRevision", "maxLength": MAX_REVISION_CHARS},
        )
    return revision


def _assert_revision(job: Mapping[str, Any], expected: str) -> None:
    current = _revision(job)
    if not hmac.compare_digest(current, expected):
        raise ScheduleManagementError(
            "revision_conflict",
            "This schedule changed after it was loaded. Reload it before saving.",
            status=409,
            details={
                "resource": "schedule",
                "resourceId": str(job.get("id") or ""),
                "expectedRevision": expected,
                "currentRevision": current,
            },
        )


def _get_native_job_locked(job_id: str) -> Dict[str, Any]:
    from cron.jobs import get_job

    job = get_job(job_id)
    if job is None:
        raise ScheduleManagementError(
            "managementNotFound",
            "Schedule not found.",
            status=404,
        )
    return job


def _job_mode(job: Mapping[str, Any]) -> str:
    if job.get("monitor_script") or job.get("monitor_url"):
        return "monitor"
    if job.get("script") or job.get("no_agent"):
        return "script"
    return "agent"


def _normalize_execution(record: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    status = _bounded_text(record.get("status") or "unknown", 32)
    return {
        "id": _bounded_text(record.get("id") or "", 128),
        "status": status,
        "source": _bounded_text(_redact_text(record.get("source") or "unknown"), 64),
        "claimedAt": _bounded_text(record.get("claimed_at") or "", 64),
        "startedAt": (
            _bounded_text(record.get("started_at"), 64)
            if record.get("started_at") is not None
            else None
        ),
        "finishedAt": (
            _bounded_text(record.get("finished_at"), 64)
            if record.get("finished_at") is not None
            else None
        ),
        "error": (
            _bounded_text(_redact_text(record.get("error")), MAX_ERROR_CHARS)
            if record.get("error")
            else None
        ),
    }


def _normalize_job(
    job: Mapping[str, Any],
    *,
    latest_execution: Optional[Mapping[str, Any]] = None,
    running_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    job_id = str(job.get("id") or "")
    mode = _job_mode(job)
    raw_name = str(job.get("name") or job_id or "Schedule")
    safe_name = _redact_text(raw_name)
    name_was_redacted = safe_name != raw_name
    name_was_truncated = len(safe_name) > MAX_NAME_CHARS
    raw_prompt = str(job.get("prompt") or "")
    prompt_was_placeholder = not bool(raw_prompt.strip())
    if prompt_was_placeholder:
        if mode == "script":
            raw_prompt = "Protected script schedule (script details are not exposed)."
        elif mode == "monitor":
            raw_prompt = "Protected monitor schedule (monitor details are not exposed)."
        else:
            raw_prompt = "Scheduled Hermes task configured by skills."
    safe_prompt = _redact_text(raw_prompt)
    prompt_was_redacted = safe_prompt != raw_prompt
    prompt_was_truncated = len(safe_prompt) > MAX_PROMPT_CHARS
    safe_prompt = _bounded_text(safe_prompt, MAX_PROMPT_CHARS)
    latest = _normalize_execution(latest_execution)
    latest_status = latest["status"] if latest else None
    running = bool(
        (running_ids and job_id in running_ids)
        or latest_status in {"claimed", "running"}
    )
    last_error = job.get("last_error") or job.get("last_delivery_error")
    if job.get("last_status"):
        last_result: Optional[Dict[str, Any]] = {
            "status": _bounded_text(job.get("last_status"), 32),
            "error": (
                _bounded_text(_redact_text(last_error), MAX_ERROR_CHARS)
                if last_error
                else None
            ),
        }
    elif latest and latest["status"] not in {"claimed", "running"}:
        # Fallback covers attempts recorded before job-level bookkeeping could
        # complete (for example an executor submission failure).
        last_result = {
            "status": latest["status"],
            "error": latest["error"],
        }
    else:
        last_result = None

    schedule = job.get("schedule") if isinstance(job.get("schedule"), Mapping) else {}
    schedule_display = str(
        job.get("schedule_display")
        or schedule.get("display")
        or schedule.get("expr")
        or schedule.get("run_at")
        or ""
    )
    skills = job.get("skills")
    if not isinstance(skills, list):
        skills = [job["skill"]] if job.get("skill") else []
    schedule_was_truncated = len(schedule_display) > MAX_SCHEDULE_CHARS
    safe_skills = [
        _redact_text(skill)
        for skill in skills
        if isinstance(skill, str) and skill.strip()
    ]
    skills_roundtrip_safe = len(skills) <= MAX_SKILLS and all(
        isinstance(skill, str)
        and 0 < len(skill.strip()) <= MAX_SKILL_CHARS
        and _redact_text(skill) == skill
        for skill in skills
    )

    # Advanced jobs remain controllable but their hidden execution fields cannot
    # be faithfully round-tripped through this intentionally narrow API.
    editable = (
        mode == "agent"
        and not name_was_redacted
        and not name_was_truncated
        and not prompt_was_placeholder
        and not prompt_was_redacted
        and not prompt_was_truncated
        and not schedule_was_truncated
        and skills_roundtrip_safe
    )
    return {
        "id": job_id,
        "name": _bounded_text(safe_name, MAX_NAME_CHARS),
        "prompt": safe_prompt,
        "schedule": _bounded_text(schedule_display, MAX_SCHEDULE_CHARS),
        "scheduleKind": _bounded_text(schedule.get("kind") or "", 32),
        "enabled": bool(job.get("enabled", True)),
        "state": _bounded_text(job.get("state") or "scheduled", 32),
        "nextRunAt": job.get("next_run_at") or None,
        "lastRunAt": job.get("last_run_at") or None,
        "lastStatus": (
            _bounded_text(job.get("last_status"), 32)
            if job.get("last_status") is not None
            else None
        ),
        "lastResult": last_result,
        "running": running,
        "skills": [
            _bounded_text(skill, MAX_SKILL_CHARS)
            for skill in safe_skills[:MAX_SKILLS]
        ],
        "mode": mode,
        "editable": editable,
        "revision": _revision(job),
    }


def _load_native_jobs() -> tuple[Path, List[Dict[str, Any]]]:
    from cron.jobs import _normalize_job_record, load_jobs

    with _job_store() as home:
        jobs = [_normalize_job_record(job) for job in load_jobs()]
    return home, jobs


def _running_job_ids() -> set[str]:
    try:
        from cron.scheduler import get_running_job_ids

        return set(get_running_job_ids())
    except Exception:
        logger.debug("Could not read cron running set", exc_info=True)
        return set()


def list_schedules() -> Dict[str, Any]:
    """List every schedule in the active profile, including disabled jobs."""
    home, jobs = _load_native_jobs()
    try:
        with _execution_store(home) as executions:
            latest = executions.latest_executions(
                [str(job.get("id") or "") for job in jobs]
            )
    except Exception as exc:
        logger.warning("Failed to read schedule execution summaries: %s", exc)
        latest = {}
    running_ids = _running_job_ids()
    schedules = [
        _normalize_job(
            job,
            latest_execution=latest.get(str(job.get("id") or "")),
            running_ids=running_ids,
        )
        for job in jobs
    ]
    return {"schedules": schedules, "count": len(schedules)}


def get_schedule(job_id: str) -> Dict[str, Any]:
    canonical_id = _validate_job_id(job_id)
    with _job_store() as home:
        job = _get_native_job_locked(canonical_id)
    latest = None
    try:
        with _execution_store(home) as executions:
            latest = executions.latest_execution(canonical_id)
    except Exception as exc:
        logger.warning("Failed to read execution summary for schedule %s: %s", canonical_id, exc)
    return {
        "schedule": _normalize_job(
            job,
            latest_execution=latest,
            running_ids=_running_job_ids(),
        )
    }


def validate_schedule(payload: Mapping[str, Any]) -> Dict[str, Any]:
    body = _require_payload(payload)
    _reject_unknown_fields(body, frozenset({"schedule"}))
    schedule = _required_text(body, "schedule", MAX_SCHEDULE_CHARS)
    normalized = _parse_schedule(schedule)
    normalized.pop("parsed", None)
    return {"valid": True, "normalized": normalized}


def create_schedule(payload: Mapping[str, Any]) -> Dict[str, Any]:
    body = _require_payload(payload)
    _reject_unknown_fields(body, _CREATE_FIELDS)
    name = _required_text(body, "name", MAX_NAME_CHARS)
    if _redact_text(name) != name:
        raise ScheduleManagementError(
            "managementUnsafeResource",
            "The schedule name contains sensitive data.",
            status=422,
            details={"field": "name"},
        )
    prompt = _required_text(body, "prompt", MAX_PROMPT_CHARS)
    schedule = _required_text(body, "schedule", MAX_SCHEDULE_CHARS)
    enabled = _optional_enabled(body)
    enabled = True if enabled is None else enabled
    skills = _normalize_skills(body.get("skills"), present="skills" in body) or []
    _scan_prompt(prompt)
    _parse_schedule(schedule)

    from cron.jobs import _jobs_lock, create_job, pause_job
    from cron.scheduler import (
        CronSchedulerRegistrationError,
        create_job_with_scheduler_registration,
    )

    try:
        with _job_store():
            if enabled:
                job = create_job_with_scheduler_registration(
                    prompt=prompt,
                    schedule=schedule,
                    name=name,
                    skills=skills,
                    deliver="local",
                )
            else:
                # Keep a disabled job invisible to the provider until its pause
                # marker is durable; this avoids registering a trigger and then
                # racing to cancel it.
                with _jobs_lock():
                    job = create_job(
                        prompt=prompt,
                        schedule=schedule,
                        name=name,
                        skills=skills,
                        deliver="local",
                    )
                    job = pause_job(
                        job["id"], reason="Disabled from Hermes Chat Control Center"
                    )
    except CronSchedulerRegistrationError as exc:
        raise ScheduleManagementError(
            "managementUpstreamUnavailable",
            exc.user_message(),
            status=424,
            details={
                "scheduleId": exc.job.get("id"),
                "jobSaved": True,
                "schedulerRegistered": False,
                "retryCreate": False,
            },
        ) from exc
    except ScheduleManagementError:
        raise
    except ValueError as exc:
        raise ScheduleManagementError(
            "managementValidationFailed",
            str(exc),
            status=422,
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create schedule")
        raise ScheduleManagementError(
            "managementUpstreamUnavailable",
            "The schedule could not be saved.",
            status=503,
        ) from exc

    if not enabled:
        _notify_jobs_changed()
    return get_schedule(str(job["id"]))


def _content_updates(body: Mapping[str, Any]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    name = _optional_text(body, "name", MAX_NAME_CHARS)
    if name is not None:
        if _redact_text(name) != name:
            raise ScheduleManagementError(
                "managementUnsafeResource",
                "The schedule name contains sensitive data.",
                status=422,
                details={"field": "name"},
            )
        updates["name"] = name
    prompt = _optional_text(body, "prompt", MAX_PROMPT_CHARS)
    if prompt is not None:
        _scan_prompt(prompt)
        updates["prompt"] = prompt
    schedule = _optional_text(body, "schedule", MAX_SCHEDULE_CHARS)
    if schedule is not None:
        _parse_schedule(schedule)
        updates["schedule"] = schedule
    skills = _normalize_skills(body.get("skills"), present="skills" in body)
    if skills is not None:
        updates["skills"] = skills
    return updates


def update_schedule(job_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    canonical_id = _validate_job_id(job_id)
    body = _require_payload(payload)
    _reject_unknown_fields(body, _UPDATE_FIELDS)
    expected_revision = _request_revision(body)
    enabled = _optional_enabled(body)
    updates = _content_updates(body)

    from cron.jobs import _jobs_lock, pause_job, resume_job, update_job

    changed = False
    try:
        with _job_store():
            with _jobs_lock():
                current = _get_native_job_locked(canonical_id)
                _assert_revision(current, expected_revision)
                if updates and _job_mode(current) != "agent":
                    raise ScheduleManagementError(
                        "managementUnsupported",
                        "Script and monitor schedules can only be enabled, disabled, "
                        "run, or deleted here.",
                        status=409,
                    )
                if "schedule" in updates:
                    new_schedule = _parse_schedule(str(updates["schedule"]))["parsed"]
                    old_schedule = current.get("schedule") or {}
                    old_kind = (
                        old_schedule.get("kind")
                        if isinstance(old_schedule, Mapping)
                        else None
                    )
                    new_kind = new_schedule.get("kind")
                    if old_kind != new_kind:
                        updates["repeat"] = {
                            "times": 1 if new_kind == "once" else None,
                            "completed": 0,
                        }
                    elif current.get("state") in {"completed", "error"}:
                        repeat_state = dict(current.get("repeat") or {})
                        repeat_state["completed"] = 0
                        updates["repeat"] = repeat_state
                    # Match the official cron tool: editing the schedule
                    # re-arms terminal jobs, while an explicitly paused job
                    # stays paused until the user enables it.
                    if current.get("state") != "paused":
                        updates["state"] = "scheduled"
                        updates["enabled"] = True
                if updates:
                    updated = update_job(canonical_id, updates)
                    if updated is None:
                        raise ScheduleManagementError(
                            "managementNotFound", "Schedule not found.", status=404
                        )
                    current = updated
                    changed = True
                if enabled is not None and enabled != bool(current.get("enabled", True)):
                    updated = (
                        resume_job(canonical_id)
                        if enabled
                        else pause_job(
                            canonical_id,
                            reason="Disabled from Hermes Chat Control Center",
                        )
                    )
                    if updated is None:
                        raise ScheduleManagementError(
                            "managementNotFound", "Schedule not found.", status=404
                        )
                    changed = True
    except ScheduleManagementError:
        raise
    except ValueError as exc:
        raise ScheduleManagementError(
            "managementValidationFailed",
            str(exc),
            status=422,
        ) from exc
    except Exception as exc:
        logger.exception("Failed to update schedule %s", canonical_id)
        raise ScheduleManagementError(
            "managementUpstreamUnavailable",
            "The schedule could not be updated.",
            status=503,
        ) from exc

    if changed:
        _notify_jobs_changed()
    return get_schedule(canonical_id)


def delete_schedule(
    job_id: str, payload: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    canonical_id = _validate_job_id(job_id)
    body = _require_payload({} if payload is None else payload)
    _reject_unknown_fields(body, frozenset({"revision", "expectedRevision"}))
    expected_revision = _request_revision(body)

    from cron.jobs import _jobs_lock, remove_job

    try:
        with _job_store():
            with _jobs_lock():
                current = _get_native_job_locked(canonical_id)
                _assert_revision(current, expected_revision)
                if not remove_job(canonical_id):
                    raise ScheduleManagementError(
                        "managementNotFound", "Schedule not found.", status=404
                    )
    except ScheduleManagementError:
        raise
    except Exception as exc:
        logger.exception("Failed to delete schedule %s", canonical_id)
        raise ScheduleManagementError(
            "managementUpstreamUnavailable",
            "The schedule could not be deleted.",
            status=503,
        ) from exc

    _notify_jobs_changed()
    return {"deleted": True, "id": canonical_id}


def run_schedule_now(
    job_id: str, payload: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Validate and accept a run-now request before background dispatch.

    The HTTP adapter calls this off-loop, returns its accepted envelope, and
    dispatches :func:`execute_schedule_now` with ``asyncio.to_thread``.  Keeping
    validation separate prevents a minutes-long agent run from holding open
    the management request while preserving the request's profile ContextVar.
    """
    canonical_id = _validate_job_id(job_id)
    body = _require_payload({} if payload is None else payload)
    _reject_unknown_fields(body, frozenset({"revision", "expectedRevision"}))
    expected_revision = _request_revision(body)

    from cron.jobs import _jobs_lock, is_job_runnable

    try:
        with _job_store():
            with _jobs_lock():
                current = _get_native_job_locked(canonical_id)
                _assert_revision(current, expected_revision)
                if not is_job_runnable(current):
                    raise ScheduleManagementError(
                        "managementConflict",
                        "Enable this schedule before running it.",
                        status=409,
                    )
                if current.get("state") == "completed":
                    raise ScheduleManagementError(
                        "managementConflict",
                        "This one-time schedule has already completed.",
                        status=409,
                    )
                if canonical_id in _running_job_ids():
                    raise ScheduleManagementError(
                        "managementConflict",
                        "This schedule is already running.",
                        status=409,
                    )
    except ScheduleManagementError:
        raise
    except Exception as exc:
        logger.exception("Failed to validate run for schedule %s", canonical_id)
        raise ScheduleManagementError(
            "managementUpstreamUnavailable",
            "The schedule could not be started.",
            status=503,
        ) from exc

    response = get_schedule(canonical_id)
    return {
        "accepted": True,
        "executionStatus": "queued",
        "schedule": response["schedule"],
    }


def execute_schedule_now(
    job_id: str,
    adapters: Any = None,
    loop: Any = None,
) -> Dict[str, Any]:
    """Run one accepted schedule through Hermes' shared firing pipeline.

    This function is blocking by design and must run in ``asyncio.to_thread``.
    The scheduler provider performs the durable fire claim, advances recurring
    schedules, creates the execution-ledger attempt, and calls the canonical
    execute/save/deliver/mark body.  The process-local running set closes the
    duplicate-run gap for executions that outlive the store claim TTL.
    """
    canonical_id = _validate_job_id(job_id)
    from cron.jobs import is_job_runnable
    from cron.scheduler import release_running_job, try_register_running_job
    from cron.scheduler_provider import resolve_cron_scheduler

    with _job_store():
        current = _get_native_job_locked(canonical_id)
        if not is_job_runnable(current):
            raise ScheduleManagementError(
                "managementConflict",
                "This schedule was disabled before the run started.",
                status=409,
            )
        if current.get("state") == "completed":
            raise ScheduleManagementError(
                "managementConflict",
                "This one-time schedule has already completed.",
                status=409,
            )
        if not try_register_running_job(canonical_id):
            raise ScheduleManagementError(
                "managementConflict",
                "This schedule is already running.",
                status=409,
            )
        try:
            try:
                processed = resolve_cron_scheduler().fire_due(
                    canonical_id,
                    adapters=adapters,
                    loop=loop,
                )
            except ScheduleManagementError:
                raise
            except Exception as exc:
                logger.exception("Immediate schedule execution failed for %s", canonical_id)
                raise ScheduleManagementError(
                    "managementUpstreamUnavailable",
                    "The scheduler could not execute this run.",
                    status=503,
                ) from exc
        finally:
            release_running_job(canonical_id)

    if not processed:
        raise ScheduleManagementError(
            "managementConflict",
            "The scheduler could not claim this run; it may already be in progress.",
            status=409,
        )
    return {"executed": True, "id": canonical_id}


def list_schedule_history(job_id: str, limit: int = 20) -> Dict[str, Any]:
    canonical_id = _validate_job_id(job_id)
    try:
        normalized_limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ScheduleManagementError(
            "managementValidationFailed",
            "History limit must be an integer.",
        ) from exc
    normalized_limit = max(1, min(normalized_limit, MAX_HISTORY_LIMIT))

    with _job_store() as home:
        _get_native_job_locked(canonical_id)
    try:
        with _execution_store(home) as executions:
            records = executions.list_executions(
                job_id=canonical_id,
                limit=normalized_limit,
            )
    except Exception as exc:
        logger.exception("Failed to read schedule history for %s", canonical_id)
        raise ScheduleManagementError(
            "managementUpstreamUnavailable",
            "Schedule history is temporarily unavailable.",
            status=503,
        ) from exc

    history = [
        normalized
        for record in records
        if (normalized := _normalize_execution(record)) is not None
    ]
    return {
        "jobId": canonical_id,
        "history": history,
        "limit": normalized_limit,
    }


def _notify_jobs_changed() -> None:
    """Best-effort provider reconciliation after a durable store mutation."""
    try:
        from cron.scheduler import _notify_provider_jobs_changed

        _notify_provider_jobs_changed()
    except Exception:
        logger.debug("Schedule provider reconciliation failed", exc_info=True)


__all__ = [
    "ScheduleManagementError",
    "create_schedule",
    "delete_schedule",
    "execute_schedule_now",
    "get_schedule",
    "list_schedule_history",
    "list_schedules",
    "run_schedule_now",
    "update_schedule",
    "validate_schedule",
]

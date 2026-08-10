"""Safe management primitives for the selected Hermes profile.

The API-server route layer intentionally stays thin.  This module owns the
filesystem and validation contract for the active ``HERMES_HOME`` only:

* ``config.yaml`` is read from and written to the selected profile;
* assistant documents are restricted to Hermes' three real identity/memory
  files; and
* every mutation is revision-checked, backed up, and atomically replaced.

No function accepts a profile name or filesystem path.  Callers therefore
cannot escape the profile selected before the gateway process was imported.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from hermes_constants import get_hermes_home
from hermes_cli.config_defaults import DEFAULT_CONFIG


MAX_CONFIGURATION_BYTES = 1024 * 1024
MAX_DOCUMENT_BYTES = 512 * 1024
REDACTED_VALUE = "__HERMES_REDACTED__"

_MISSING = object()
_TREE_MAX_DEPTH = 40
_TREE_MAX_NODES = 20_000
_MANAGEMENT_THREAD_LOCK = threading.RLock()
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")


class ManagementProfileError(Exception):
    """Normalized error consumed by the async API route layer."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


class ManagementValidationError(ManagementProfileError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            "validation_failed", message, status_code=422, details=details
        )


class ManagementConflictError(ManagementProfileError):
    def __init__(self, current_revision: str) -> None:
        super().__init__(
            "revision_conflict",
            "The resource changed after it was loaded.",
            status_code=409,
            details={"currentRevision": current_revision},
        )


class ManagementNotFoundError(ManagementProfileError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, status_code=404)


@dataclass(frozen=True)
class AssistantDocumentSpec:
    id: str
    relative_path: tuple[str, ...]
    filename: str
    title: str
    purpose: str
    create_mode: int


# These are the canonical files used by prompt_builder, memory_tool, profile
# clone/export, and the profile bootstrapper.  Keeping the allowlist here makes
# it the authoritative management surface without accepting arbitrary paths.
ASSISTANT_DOCUMENTS: tuple[AssistantDocumentSpec, ...] = (
    AssistantDocumentSpec(
        id="soul",
        relative_path=("SOUL.md",),
        filename="SOUL.md",
        title="Identity",
        purpose="Defines the assistant's primary identity and behavior.",
        create_mode=0o644,
    ),
    AssistantDocumentSpec(
        id="user",
        relative_path=("memories", "USER.md"),
        filename="USER.md",
        title="User profile",
        purpose="Stores durable facts and preferences about the user.",
        create_mode=0o600,
    ),
    AssistantDocumentSpec(
        id="memory",
        relative_path=("memories", "MEMORY.md"),
        filename="MEMORY.md",
        title="Assistant memory",
        purpose="Stores the assistant's curated long-term notes.",
        create_mode=0o600,
    ),
)

_DOCUMENT_BY_ID = {item.id: item for item in ASSISTANT_DOCUMENTS}

# The editor highlights a deliberately small set of stable, common leaves.
# Every entry is looked up in DEFAULT_CONFIG at response time and silently
# omitted if Hermes removes or reshapes it, so the advertised schema remains
# derived from the running agent rather than becoming an independent contract.
_COMMON_CONFIGURATION_PATHS: tuple[str, ...] = (
    "agent.max_turns",
    "terminal.backend",
    "terminal.cwd",
    "compression.enabled",
    "compression.threshold",
    "display.interface",
    "display.compact",
    "display.streaming",
    "display.show_reasoning",
    "display.timestamps",
    "memory.memory_enabled",
    "memory.user_profile_enabled",
    "memory.write_approval",
    "delegation.max_concurrent_children",
    "delegation.max_spawn_depth",
    "security.redact_secrets",
    "security.allow_private_urls",
)

_SECRET_EXACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "passwd",
        "private_key",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "auth_token",
        "bot_token",
        "bearer_token",
        "client_secret",
        "signing_secret",
        "webhook_secret",
        "sudo_password",
        "authorization",
        "cookie",
        "cookies",
        "credential",
        "credentials",
    }
)
_SECRET_CONTAINER_KEYS = frozenset(
    {"auth", "authentication", "headers", "http_headers", "extra_headers"}
)
_NON_SECRET_TOKEN_MARKERS = (
    "max_",
    "min_",
    "limit",
    "budget",
    "count",
    "threshold",
    "context",
    "input",
    "output",
    "usage",
    "tokenizer",
    "tokens_",
)


def _error(
    code: str,
    message: str,
    *,
    status_code: int = 400,
    details: Mapping[str, Any] | None = None,
) -> ManagementProfileError:
    return ManagementProfileError(
        code, message, status_code=status_code, details=details
    )


def _selected_home() -> Path:
    configured = get_hermes_home().expanduser()
    try:
        home = configured.resolve(strict=True)
    except OSError as exc:
        raise ManagementNotFoundError(
            "profile_not_found", "The selected Hermes profile does not exist."
        ) from exc
    if not home.is_dir():
        raise ManagementNotFoundError(
            "profile_not_found", "The selected Hermes profile is not a directory."
        )
    return home


def _configuration_write_allowed() -> bool:
    """Honor Hermes' package-manager ownership of declarative config."""

    try:
        from hermes_cli.config import is_managed

        return not is_managed()
    except Exception:
        # Capability/permission checks fail closed.  A broken authority probe
        # must never turn into an alternate path around managed ownership.
        return False


def _require_configuration_write_allowed() -> None:
    if not _configuration_write_allowed():
        raise _error(
            "managed_configuration",
            "Profile configuration is owned by the managed Hermes installation.",
            status_code=403,
        )


def get_management_profile_capabilities() -> dict[str, bool]:
    """Return per-operation permission hints for capability advertisement."""

    try:
        home = _selected_home()
    except ManagementProfileError:
        return {
            "profileConfigurationRead": False,
            "profileConfigurationWrite": False,
            "assistantDocumentsRead": False,
            "assistantDocumentsWrite": False,
        }
    readable = os.access(home, os.R_OK)
    writable = os.access(home, os.W_OK)
    return {
        "profileConfigurationRead": readable,
        "profileConfigurationWrite": writable and _configuration_write_allowed(),
        "assistantDocumentsRead": readable,
        "assistantDocumentsWrite": writable,
    }


def _ensure_directory(home: Path, parts: Sequence[str], *, create: bool) -> Path:
    current = home
    for part in parts:
        if not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise _error("unsafe_path", "Unsafe profile resource path.", status_code=403)
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                return current
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise _error(
                    "resource_unwritable",
                    "A profile resource directory cannot be created.",
                    status_code=500,
                ) from exc
            info = current.lstat()
        except OSError as exc:
            raise _error(
                "resource_unreadable",
                "A profile resource directory cannot be inspected.",
                status_code=500,
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise _error(
                "unsafe_symlink",
                "Symlinks are not permitted for managed profile resources.",
                status_code=403,
            )
        if not stat.S_ISDIR(info.st_mode):
            raise _error(
                "unsafe_path",
                "A managed profile resource parent is not a directory.",
                status_code=403,
            )
        try:
            current.resolve(strict=True).relative_to(home)
        except (OSError, ValueError) as exc:
            raise _error(
                "path_escape",
                "A managed profile resource escapes the selected profile.",
                status_code=403,
            ) from exc
    return current


def _resource_path(
    home: Path, relative_path: Sequence[str], *, create_parent: bool = False
) -> Path:
    if not relative_path:
        raise _error("unsafe_path", "Unsafe profile resource path.", status_code=403)
    parent = _ensure_directory(home, relative_path[:-1], create=create_parent)
    filename = relative_path[-1]
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise _error("unsafe_path", "Unsafe profile resource path.", status_code=403)
    target = parent / filename
    if os.path.lexists(target):
        try:
            info = target.lstat()
        except OSError as exc:
            raise _error(
                "resource_unreadable",
                "A managed profile resource cannot be inspected.",
                status_code=500,
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise _error(
                "unsafe_symlink",
                "Symlinks are not permitted for managed profile resources.",
                status_code=403,
            )
        if not stat.S_ISREG(info.st_mode):
            raise _error(
                "unsafe_path",
                "A managed profile resource is not a regular file.",
                status_code=403,
            )
    try:
        # A document such as memories/USER.md is a supported resource even
        # before its parent has been created.  ``strict=False`` still resolves
        # every existing component (which was lstat-checked above) while
        # allowing that honest missing-file state.
        parent.resolve(strict=False).relative_to(home)
    except (OSError, ValueError) as exc:
        raise _error(
            "path_escape",
            "A managed profile resource escapes the selected profile.",
            status_code=403,
        ) from exc
    return target


def _read_resource(path: Path, *, max_bytes: int) -> tuple[bool, bytes]:
    if not os.path.lexists(path):
        return False, b""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return False, b""
    except OSError as exc:
        if os.path.islink(path):
            raise _error(
                "unsafe_symlink",
                "Symlinks are not permitted for managed profile resources.",
                status_code=403,
            ) from exc
        raise _error(
            "resource_unreadable",
            "The managed profile resource cannot be read.",
            status_code=500,
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _error(
                "unsafe_path",
                "A managed profile resource is not a regular file.",
                status_code=403,
            )
        if info.st_size > max_bytes:
            raise _error(
                "resource_too_large",
                "The managed profile resource exceeds the size limit.",
                status_code=413,
                details={"sizeBytes": info.st_size, "maxBytes": max_bytes},
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise _error(
                "resource_too_large",
                "The managed profile resource exceeds the size limit.",
                status_code=413,
                details={"sizeBytes": len(data), "maxBytes": max_bytes},
            )
        return True, data
    finally:
        os.close(fd)


def _revision(data: bytes, *, exists: bool) -> str:
    prefix = b"exists\0" if exists else b"missing\0"
    return "sha256:" + hashlib.sha256(prefix + data).hexdigest()


def _require_revision(expected_revision: str, current_revision: str) -> None:
    if not isinstance(expected_revision, str) or not expected_revision.strip():
        raise _error(
            "precondition_required",
            "expectedRevision is required for this mutation.",
            status_code=428,
            details={"currentRevision": current_revision},
        )
    if expected_revision != current_revision:
        raise ManagementConflictError(current_revision)


def _require_resource_unchanged(
    path: Path, *, expected_revision: str, max_bytes: int
) -> None:
    """Recheck after validation/backup to narrow the external-writer window."""

    exists, data = _read_resource(path, max_bytes=max_bytes)
    current_revision = _revision(data, exists=exists)
    if current_revision != expected_revision:
        raise ManagementConflictError(current_revision)


def _atomic_write(path: Path, data: bytes, *, create_mode: int) -> None:
    """Atomically replace a verified non-symlink resource."""

    parent = path.parent
    original_mode: int | None = None
    original_owner: tuple[int, int] | None = None
    if os.path.lexists(path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise _error(
                "unsafe_symlink" if stat.S_ISLNK(info.st_mode) else "unsafe_path",
                "The managed profile resource cannot be atomically replaced.",
                status_code=403,
            )
        original_mode = stat.S_IMODE(info.st_mode)
        original_owner = (info.st_uid, info.st_gid)
    effective_mode = original_mode if original_mode is not None else create_mode

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, effective_mode)
        if original_owner is not None and hasattr(os, "fchown"):
            try:
                os.fchown(fd, *original_owner)
            except OSError:
                pass
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        # Recheck immediately before rename.  os.replace targets the directory
        # entry itself, so even a race that introduces a symlink cannot cause a
        # write through that link to an out-of-profile target.
        if os.path.lexists(path) and stat.S_ISLNK(path.lstat().st_mode):
            raise _error(
                "unsafe_symlink",
                "Symlinks are not permitted for managed profile resources.",
                status_code=403,
            )
        os.replace(temporary, path)
        temporary = ""
        if not hasattr(os, "fchmod"):
            try:
                os.chmod(path, effective_mode)
            except OSError:
                pass
        if hasattr(os, "O_DIRECTORY"):
            try:
                dir_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
    except ManagementProfileError:
        raise
    except OSError as exc:
        raise _error(
            "resource_write_failed",
            "The managed profile resource could not be written.",
            status_code=500,
        ) from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


@contextmanager
def _management_write_lock(home: Path) -> Iterator[None]:
    """Serialize revision-check + backup + replace across threads/processes."""

    with _MANAGEMENT_THREAD_LOCK:
        lock_path = _resource_path(home, (".management.lock",))
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(fd, "r+b", buffering=0)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    handle.write(b"\0")
                handle.seek(0)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            handle.close()


def _backup_directory(home: Path, resource_key: str, *, create: bool) -> Path:
    return _ensure_directory(
        home, ("backups", "management", resource_key), create=create
    )


def _backup_resource(home: Path, resource_key: str, data: bytes) -> bool:
    directory = _backup_directory(home, resource_key, create=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    name = f"{stamp}-{time.time_ns()}-{digest}.bak"
    target = _resource_path(
        home, ("backups", "management", resource_key, name), create_parent=True
    )
    _atomic_write(target, data, create_mode=0o600)
    return True


def _backup_records(
    home: Path, resource_key: str, *, max_bytes: int
) -> list[tuple[Path, str, bytes]]:
    directory = _backup_directory(home, resource_key, create=False)
    if not directory.exists():
        return []
    records: list[tuple[Path, str, bytes]] = []
    try:
        candidates = sorted(directory.glob("*.bak"), reverse=True)
    except OSError:
        return []
    for candidate in candidates:
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            exists, data = _read_resource(candidate, max_bytes=max_bytes)
            if exists:
                records.append((candidate, _revision(data, exists=True), data))
        except ManagementProfileError:
            continue
    return records


def _latest_backup_metadata(
    home: Path, resource_key: str, *, max_bytes: int
) -> tuple[bool, str | None]:
    records = _backup_records(home, resource_key, max_bytes=max_bytes)
    return (bool(records), records[0][1] if records else None)


def _text_errors(content: Any, *, max_bytes: int) -> tuple[list[dict[str, Any]], bytes]:
    errors: list[dict[str, Any]] = []
    if not isinstance(content, str):
        return ([{"code": "invalid_text", "message": "Content must be text."}], b"")
    try:
        encoded = content.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return (
            [{"code": "invalid_utf8", "message": "Content must be valid UTF-8 text."}],
            b"",
        )
    if len(encoded) > max_bytes:
        errors.append(
            {
                "code": "resource_too_large",
                "message": "Content exceeds the size limit.",
                "sizeBytes": len(encoded),
                "maxBytes": max_bytes,
            }
        )
    for character in content:
        codepoint = ord(character)
        if (codepoint < 32 and character not in {"\t", "\n", "\r"}) or codepoint == 127:
            errors.append(
                {
                    "code": "invalid_text_control",
                    "message": "Content contains a non-text control character.",
                }
            )
            break
    return errors, encoded


def _decode_resource(data: bytes, *, label: str) -> str:
    try:
        content = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _error(
            "invalid_utf8",
            f"{label} is not valid UTF-8 text.",
            status_code=422,
        ) from exc
    errors, _ = _text_errors(content, max_bytes=max(len(data), 1))
    if errors:
        raise ManagementValidationError(
            f"{label} is not a supported text file.", details={"errors": errors}
        )
    return content


def _format_path(path: Sequence[str | int]) -> str:
    result = ""
    for part in path:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + part
    return result


def _validate_tree(value: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    count = 0

    def walk(node: Any, path: tuple[str | int, ...], ancestors: set[int], depth: int) -> None:
        nonlocal count
        if errors:
            return
        count += 1
        if count > _TREE_MAX_NODES:
            errors.append(
                {"code": "yaml_too_complex", "message": "Configuration has too many values."}
            )
            return
        if depth > _TREE_MAX_DEPTH:
            errors.append(
                {"code": "yaml_too_deep", "message": "Configuration is nested too deeply."}
            )
            return
        if isinstance(node, Mapping):
            identity = id(node)
            if identity in ancestors:
                errors.append(
                    {"code": "cyclic_yaml", "message": "Cyclic YAML aliases are not supported."}
                )
                return
            ancestors.add(identity)
            for key, child in node.items():
                if not isinstance(key, str):
                    errors.append(
                        {
                            "code": "invalid_yaml_key",
                            "message": "Configuration keys must be strings.",
                            "path": _format_path(path),
                        }
                    )
                    break
                walk(child, path + (key,), ancestors, depth + 1)
            ancestors.remove(identity)
            return
        if isinstance(node, list):
            identity = id(node)
            if identity in ancestors:
                errors.append(
                    {"code": "cyclic_yaml", "message": "Cyclic YAML aliases are not supported."}
                )
                return
            ancestors.add(identity)
            for index, child in enumerate(node):
                walk(child, path + (index,), ancestors, depth + 1)
            ancestors.remove(identity)
            return
        if not isinstance(node, (str, int, float, bool, type(None))):
            errors.append(
                {
                    "code": "unsupported_yaml_value",
                    "message": f"Unsupported YAML value type: {type(node).__name__}.",
                    "path": _format_path(path),
                }
            )

    walk(value, (), set(), 0)
    return errors


def _parse_configuration(content: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    errors, _ = _text_errors(content, max_bytes=MAX_CONFIGURATION_BYTES)
    if errors:
        return None, errors
    try:
        loaded = yaml.safe_load(content) if content.strip() else {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        error: dict[str, Any] = {
            "code": "invalid_yaml",
            "message": "Configuration is not valid YAML.",
        }
        if mark is not None:
            error["line"] = mark.line + 1
            error["column"] = mark.column + 1
        return None, [error]
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        return None, [
            {
                "code": "invalid_yaml_root",
                "message": "Configuration must be a YAML mapping at the top level.",
            }
        ]
    tree_errors = _validate_tree(loaded)
    return (loaded if not tree_errors else None), tree_errors


def _is_secret_key(key: str, ancestors: Sequence[str | int]) -> bool:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    ancestor_names = [str(item).lower() for item in ancestors if isinstance(item, str)]
    if not ancestors and normalized == "secrets":
        return True
    if normalized in _SECRET_EXACT_KEYS or normalized in _SECRET_CONTAINER_KEYS:
        return True
    if normalized.endswith(("_api_key", "_password", "_passwd", "_secret")):
        return True
    if normalized.endswith(("_credential", "_credentials", "_private_key")):
        return True
    if normalized.endswith(("_access_key", "_secret_access_key")):
        return True
    if normalized.endswith("_token") and not any(
        marker in normalized for marker in _NON_SECRET_TOKEN_MARKERS
    ):
        return True
    if normalized == "key" and any(
        marker in ancestor
        for ancestor in ancestor_names
        for marker in (
            "provider",
            "auth",
            "credential",
            "secret",
            "oauth",
            "api_server",
        )
    ):
        return True
    return False


def _collect_secret_entries(
    value: Any, path: tuple[str | int, ...] = ()
) -> dict[tuple[str | int, ...], Any]:
    entries: dict[tuple[str | int, ...], Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (key,)
            if _is_secret_key(key, path):
                entries[child_path] = child
            else:
                entries.update(_collect_secret_entries(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            entries.update(_collect_secret_entries(child, path + (index,)))
    return entries


def _masked_copy(value: Any) -> tuple[Any, list[str]]:
    paths: list[str] = []

    def mask(node: Any, path: tuple[str | int, ...]) -> Any:
        if isinstance(node, Mapping):
            result: dict[str, Any] = {}
            for key, child in node.items():
                child_path = path + (key,)
                if _is_secret_key(key, path):
                    result[key] = REDACTED_VALUE
                    paths.append(_format_path(child_path))
                else:
                    result[key] = mask(child, child_path)
            return result
        if isinstance(node, list):
            return [mask(child, path + (index,)) for index, child in enumerate(node)]
        return copy.deepcopy(node)

    return mask(value, ()), sorted(paths)


def _lookup_path(value: Any, path: Sequence[str | int]) -> Any:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return _MISSING
            current = current[part]
        else:
            if not isinstance(current, Mapping) or part not in current:
                return _MISSING
            current = current[part]
    return current


def _set_path(value: Any, path: Sequence[str | int], replacement: Any) -> bool:
    current = value
    for index, part in enumerate(path[:-1]):
        next_part = path[index + 1]
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return False
            current = current[part]
        else:
            if not isinstance(current, dict):
                return False
            if part not in current:
                if isinstance(next_part, int):
                    return False
                current[part] = {}
            current = current[part]
    final = path[-1]
    if isinstance(final, int):
        if not isinstance(current, list) or final >= len(current):
            return False
        current[final] = copy.deepcopy(replacement)
    else:
        if not isinstance(current, dict):
            return False
        current[final] = copy.deepcopy(replacement)
    return True


def _identity_marker(item: Any) -> tuple[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    for key in ("name", "id", "provider", "label"):
        value = item.get(key)
        if isinstance(value, (str, int)):
            return key, value
    return None


def _secret_list_guard_errors(
    current: Mapping[str, Any], candidate: Mapping[str, Any], secret_paths: Sequence[Sequence[str | int]]
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    guarded: set[tuple[str | int, ...]] = set()
    for secret_path in secret_paths:
        for position, part in enumerate(secret_path):
            if isinstance(part, int):
                guarded.add(tuple(secret_path[:position]))
    for list_path in sorted(guarded, key=_format_path):
        old_list = _lookup_path(current, list_path)
        new_list = _lookup_path(candidate, list_path)
        if not isinstance(old_list, list) or not isinstance(new_list, list):
            errors.append(
                {
                    "code": "secret_container_changed",
                    "message": "A list containing protected secrets cannot be removed or reshaped here.",
                    "path": _format_path(list_path),
                }
            )
            continue
        if len(old_list) != len(new_list):
            errors.append(
                {
                    "code": "secret_container_changed",
                    "message": "Items containing protected secrets cannot be added or removed here.",
                    "path": _format_path(list_path),
                }
            )
            continue
        for index, (old_item, new_item) in enumerate(zip(old_list, new_list)):
            marker = _identity_marker(old_item)
            if marker is not None:
                if _identity_marker(new_item) != marker:
                    errors.append(
                        {
                            "code": "secret_container_changed",
                            "message": "Items containing protected secrets cannot be reordered here.",
                            "path": _format_path(list_path + (index,)),
                        }
                    )
                    break
            else:
                old_masked, _ = _masked_copy(old_item)
                new_masked, _ = _masked_copy(new_item)
                if old_masked != new_masked:
                    errors.append(
                        {
                            "code": "secret_container_changed",
                            "message": "This anonymous item contains a protected secret and cannot be reshaped here.",
                            "path": _format_path(list_path + (index,)),
                        }
                    )
                    break
    return errors


def _preserve_secrets(
    current: Mapping[str, Any], candidate: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    current_entries = _collect_secret_entries(current)
    candidate_entries = _collect_secret_entries(candidate)
    redacted_paths = sorted(_format_path(path) for path in current_entries)

    errors.extend(
        _secret_list_guard_errors(current, candidate, tuple(current_entries.keys()))
    )
    current_paths = set(current_entries)
    for path, value in candidate_entries.items():
        if path not in current_paths:
            errors.append(
                {
                    "code": "secret_not_editable",
                    "message": "Secrets cannot be added through profile configuration management.",
                    "path": _format_path(path),
                }
            )
        elif value != REDACTED_VALUE:
            errors.append(
                {
                    "code": "secret_not_editable",
                    "message": "Protected secret values cannot be edited here.",
                    "path": _format_path(path),
                }
            )

    if errors:
        return errors, warnings, redacted_paths

    omitted = 0
    for path, value in current_entries.items():
        candidate_value = _lookup_path(candidate, path)
        if candidate_value is _MISSING:
            omitted += 1
        if not _set_path(candidate, path, value):
            errors.append(
                {
                    "code": "secret_container_changed",
                    "message": "A protected secret could not be safely preserved.",
                    "path": _format_path(path),
                }
            )
    if omitted and not errors:
        warnings.append(
            {
                "code": "secrets_preserved",
                "message": "Omitted protected secret values were preserved unchanged.",
            }
        )
    return errors, warnings, redacted_paths


def _configuration_structure_issues(
    configuration: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        from hermes_cli.config import validate_config_structure

        issues = validate_config_structure(configuration)
    except Exception:
        issues = []
    for issue in issues:
        item = {
            "code": "invalid_configuration_structure"
            if issue.severity == "error"
            else "configuration_warning",
            "message": issue.message,
            "hint": issue.hint,
        }
        (errors if issue.severity == "error" else warnings).append(item)
    return errors, warnings


def _configuration_validation(
    content: str, current: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    candidate, errors = _parse_configuration(content)
    warnings: list[dict[str, Any]] = []
    redacted_paths = sorted(
        _format_path(path) for path in _collect_secret_entries(current)
    )
    if candidate is not None:
        structure_errors, structure_warnings = _configuration_structure_issues(candidate)
        errors.extend(structure_errors)
        warnings.extend(structure_warnings)
        secret_errors, secret_warnings, redacted_paths = _preserve_secrets(
            current, candidate
        )
        errors.extend(secret_errors)
        warnings.extend(secret_warnings)
    result = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "redactedPaths": redacted_paths,
        "sizeBytes": len(content.encode("utf-8", errors="ignore"))
        if isinstance(content, str)
        else 0,
        "maxBytes": MAX_CONFIGURATION_BYTES,
    }
    if errors or candidate is None:
        return result, None, None

    if redacted_paths:
        # The client sees placeholders, never values.  Serialize the merged
        # parsed document so no placeholder can reach disk.  Secret-free files
        # keep the user's exact raw YAML and formatting.
        final_content = yaml.safe_dump(
            candidate,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    else:
        final_content = content
    return result, candidate, final_content


def _value_at(mapping: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    current: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _field_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "null"


def _configuration_fields(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    writable = _configuration_write_allowed()
    for key in _COMMON_CONFIGURATION_PATHS:
        has_default, default = _value_at(DEFAULT_CONFIG, key)
        if not has_default or isinstance(default, Mapping):
            continue
        overridden, value = _value_at(raw, key)
        fields.append(
            {
                "key": key,
                "type": _field_type(default),
                "defaultValue": copy.deepcopy(default),
                "value": copy.deepcopy(value if overridden else default),
                "source": "override" if overridden else "default",
                "category": key.split(".", 1)[0],
                "writable": writable,
            }
        )
    return fields


def _load_current_configuration(
    home: Path,
) -> tuple[Path, bool, bytes, str, dict[str, Any]]:
    path = _resource_path(home, ("config.yaml",))
    exists, data = _read_resource(path, max_bytes=MAX_CONFIGURATION_BYTES)
    content = _decode_resource(data, label="config.yaml") if exists else ""
    parsed, errors = _parse_configuration(content)
    if errors or parsed is None:
        raise ManagementValidationError(
            "The current config.yaml cannot be safely managed.",
            details={"errors": errors},
        )
    return path, exists, data, content, parsed


def _configuration_response(
    home: Path,
    *,
    exists: bool,
    data: bytes,
    content: str,
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    masked, redacted_paths = _masked_copy(parsed)
    if redacted_paths:
        public_content = yaml.safe_dump(
            masked,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    else:
        public_content = content
    _, warnings = _configuration_structure_issues(dict(parsed))
    backup_available, backup_revision = _latest_backup_metadata(
        home, "configuration", max_bytes=MAX_CONFIGURATION_BYTES
    )
    return {
        "content": public_content,
        "revision": _revision(data, exists=exists),
        "exists": exists,
        "format": "yaml",
        "sizeBytes": len(data),
        "maxBytes": MAX_CONFIGURATION_BYTES,
        "redactedPaths": redacted_paths,
        "fields": _configuration_fields(parsed),
        "warnings": warnings,
        "backupAvailable": backup_available,
        "backupRevision": backup_revision,
        "writable": _configuration_write_allowed(),
    }


def get_profile_configuration() -> dict[str, Any]:
    """Return the selected profile's authoritative, secret-redacted config."""

    home = _selected_home()
    _, exists, data, content, parsed = _load_current_configuration(home)
    return _configuration_response(
        home, exists=exists, data=data, content=content, parsed=parsed
    )


def validate_profile_configuration(content: str) -> dict[str, Any]:
    """Validate a redacted raw YAML draft without writing it."""

    _require_configuration_write_allowed()
    home = _selected_home()
    _, _, _, _, current = _load_current_configuration(home)
    result, _, _ = _configuration_validation(content, current)
    return result


def update_profile_configuration(
    content: str, expected_revision: str
) -> dict[str, Any]:
    """Revision-check, validate, back up, and atomically replace config.yaml."""

    _require_configuration_write_allowed()
    home = _selected_home()
    with _management_write_lock(home):
        path, exists, data, _, current = _load_current_configuration(home)
        current_revision = _revision(data, exists=exists)
        _require_revision(expected_revision, current_revision)
        validation, parsed, final_content = _configuration_validation(content, current)
        if not validation["valid"] or parsed is None or final_content is None:
            raise ManagementValidationError(
                "Profile configuration validation failed.", details=validation
            )
        final_bytes = final_content.encode("utf-8")
        if len(final_bytes) > MAX_CONFIGURATION_BYTES:
            raise ManagementValidationError(
                "Profile configuration exceeds the size limit.",
                details={
                    "errors": [
                        {
                            "code": "resource_too_large",
                            "sizeBytes": len(final_bytes),
                            "maxBytes": MAX_CONFIGURATION_BYTES,
                        }
                    ]
                },
            )
        backup_created = _backup_resource(home, "configuration", data) if exists else False
        _require_resource_unchanged(
            path,
            expected_revision=current_revision,
            max_bytes=MAX_CONFIGURATION_BYTES,
        )
        _atomic_write(path, final_bytes, create_mode=0o600)
        response = _configuration_response(
            home,
            exists=True,
            data=final_bytes,
            content=final_content,
            parsed=parsed,
        )
        # Preserve validation-time feedback such as the explicit notice that
        # omitted redacted values were restored.  Rebuilding the resource
        # response above intentionally reparses the saved file, but that alone
        # cannot reconstruct draft-specific warnings.
        response.update(
            {
                "saved": True,
                "backupCreated": backup_created,
                "warnings": validation["warnings"],
            }
        )
        return response


def _document_spec(document_id: str) -> AssistantDocumentSpec:
    if not isinstance(document_id, str) or document_id not in _DOCUMENT_BY_ID:
        raise ManagementNotFoundError(
            "document_not_found", "The requested assistant document is not supported."
        )
    return _DOCUMENT_BY_ID[document_id]


def _line_ending_metadata(content: str) -> tuple[str, bool]:
    crlf = content.count("\r\n")
    without_crlf = content.replace("\r\n", "")
    lf = without_crlf.count("\n")
    cr = without_crlf.count("\r")
    styles = [name for name, count in (("crlf", crlf), ("lf", lf), ("cr", cr)) if count]
    line_ending = styles[0] if len(styles) == 1 else ("mixed" if styles else "none")
    return line_ending, content.endswith(("\n", "\r"))


def _apply_document_conventions(
    content: str,
    *,
    current_content: str,
    line_ending: str | None,
    final_newline: bool | None,
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    valid_styles = {None, "preserve", "lf", "crlf", "cr"}
    if not isinstance(line_ending, (str, type(None))) or line_ending not in valid_styles:
        errors.append(
            {
                "code": "invalid_line_ending",
                "message": "lineEnding must be preserve, lf, crlf, or cr.",
            }
        )
    if final_newline is not None and not isinstance(final_newline, bool):
        errors.append(
            {
                "code": "invalid_final_newline",
                "message": "finalNewline must be a boolean when provided.",
            }
        )
    if errors:
        return None, errors, warnings

    current_style, current_final = _line_ending_metadata(current_content)
    requested_style = line_ending
    if requested_style in {None, "preserve"}:
        if current_style in {"lf", "crlf", "cr"}:
            requested_style = current_style
        else:
            draft_style, _ = _line_ending_metadata(content)
            requested_style = draft_style if draft_style in {"lf", "crlf", "cr"} else "lf"
        if current_style == "mixed":
            warnings.append(
                {
                    "code": "mixed_line_endings",
                    "message": "The original document has mixed line endings; edited lines use LF unless explicitly selected.",
                }
            )
    separator = {"lf": "\n", "crlf": "\r\n", "cr": "\r"}[requested_style]
    normalized = _LINE_BREAK_RE.sub(separator, content)

    desired_final = final_newline
    if desired_final is None and current_content:
        desired_final = current_final
    if desired_final is True and not normalized.endswith(("\n", "\r")):
        normalized += separator
    elif desired_final is False:
        normalized = re.sub(r"(?:\r\n|\r|\n)+$", "", normalized)
    return normalized, errors, warnings


def _load_document(
    home: Path, spec: AssistantDocumentSpec
) -> tuple[Path, bool, bytes, str]:
    path = _resource_path(home, spec.relative_path)
    exists, data = _read_resource(path, max_bytes=MAX_DOCUMENT_BYTES)
    content = _decode_resource(data, label=spec.filename) if exists else ""
    return path, exists, data, content


def _document_response(
    home: Path,
    spec: AssistantDocumentSpec,
    *,
    exists: bool,
    data: bytes,
    content: str,
    include_content: bool,
) -> dict[str, Any]:
    line_ending, final_newline = _line_ending_metadata(content)
    backup_available, backup_revision = _latest_backup_metadata(
        home, f"document-{spec.id}", max_bytes=MAX_DOCUMENT_BYTES
    )
    result: dict[str, Any] = {
        "id": spec.id,
        "filename": spec.filename,
        "title": spec.title,
        "purpose": spec.purpose,
        "exists": exists,
        "revision": _revision(data, exists=exists),
        "sizeBytes": len(data),
        "maxBytes": MAX_DOCUMENT_BYTES,
        "lineEnding": line_ending,
        "finalNewline": final_newline,
        "backupAvailable": backup_available,
        "backupRevision": backup_revision,
    }
    if include_content:
        result["content"] = content
    return result


def list_assistant_documents() -> dict[str, Any]:
    """List only Hermes' supported assistant Markdown documents."""

    home = _selected_home()
    documents = []
    for spec in ASSISTANT_DOCUMENTS:
        _, exists, data, content = _load_document(home, spec)
        documents.append(
            _document_response(
                home,
                spec,
                exists=exists,
                data=data,
                content=content,
                include_content=False,
            )
        )
    return {"documents": documents}


def get_assistant_document(document_id: str) -> dict[str, Any]:
    """Read one allowlisted document on demand."""

    home = _selected_home()
    spec = _document_spec(document_id)
    _, exists, data, content = _load_document(home, spec)
    return _document_response(
        home,
        spec,
        exists=exists,
        data=data,
        content=content,
        include_content=True,
    )


def validate_assistant_document(
    document_id: str,
    content: str,
    *,
    line_ending: str | None = None,
    final_newline: bool | None = None,
) -> dict[str, Any]:
    """Validate and preview the exact text conventions a save would use."""

    home = _selected_home()
    spec = _document_spec(document_id)
    _, _, _, current_content = _load_document(home, spec)
    text_errors, _ = _text_errors(content, max_bytes=MAX_DOCUMENT_BYTES)
    normalized: str | None = None
    convention_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not text_errors:
        normalized, convention_errors, warnings = _apply_document_conventions(
            content,
            current_content=current_content,
            line_ending=line_ending,
            final_newline=final_newline,
        )
    errors = text_errors + convention_errors
    if normalized is not None:
        normalized_errors, encoded = _text_errors(
            normalized, max_bytes=MAX_DOCUMENT_BYTES
        )
        errors.extend(normalized_errors)
    else:
        encoded = b""
    resulting_style, resulting_final = _line_ending_metadata(normalized or "")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "sizeBytes": len(encoded),
        "maxBytes": MAX_DOCUMENT_BYTES,
        "lineEnding": resulting_style,
        "finalNewline": resulting_final,
    }


def update_assistant_document(
    document_id: str,
    content: str,
    expected_revision: str,
    *,
    line_ending: str | None = None,
    final_newline: bool | None = None,
) -> dict[str, Any]:
    """Revision-check, back up, and atomically replace an allowlisted file."""

    home = _selected_home()
    spec = _document_spec(document_id)
    with _management_write_lock(home):
        path, exists, data, current_content = _load_document(home, spec)
        current_revision = _revision(data, exists=exists)
        _require_revision(expected_revision, current_revision)
        validation = validate_assistant_document(
            document_id,
            content,
            line_ending=line_ending,
            final_newline=final_newline,
        )
        if not validation["valid"]:
            raise ManagementValidationError(
                f"{spec.filename} validation failed.", details=validation
            )
        normalized, _, _ = _apply_document_conventions(
            content,
            current_content=current_content,
            line_ending=line_ending,
            final_newline=final_newline,
        )
        assert normalized is not None
        final_bytes = normalized.encode("utf-8")
        backup_created = (
            _backup_resource(home, f"document-{spec.id}", data) if exists else False
        )
        path = _resource_path(home, spec.relative_path, create_parent=True)
        _require_resource_unchanged(
            path,
            expected_revision=current_revision,
            max_bytes=MAX_DOCUMENT_BYTES,
        )
        _atomic_write(path, final_bytes, create_mode=spec.create_mode)
        response = _document_response(
            home,
            spec,
            exists=True,
            data=final_bytes,
            content=normalized,
            include_content=True,
        )
        response.update({"saved": True, "backupCreated": backup_created})
        return response


def revert_assistant_document(
    document_id: str,
    expected_revision: str,
    *,
    backup_revision: str | None = None,
) -> dict[str, Any]:
    """Restore the newest (or requested) recoverable backup."""

    home = _selected_home()
    spec = _document_spec(document_id)
    with _management_write_lock(home):
        path, exists, data, _ = _load_document(home, spec)
        current_revision = _revision(data, exists=exists)
        _require_revision(expected_revision, current_revision)
        records = _backup_records(
            home, f"document-{spec.id}", max_bytes=MAX_DOCUMENT_BYTES
        )
        selected: tuple[Path, str, bytes] | None = None
        if backup_revision is None:
            selected = records[0] if records else None
        else:
            selected = next(
                (record for record in records if record[1] == backup_revision), None
            )
        if selected is None:
            raise ManagementNotFoundError(
                "backup_not_found",
                "No matching recoverable backup is available for this document.",
            )
        _, restored_revision, restored_data = selected
        restored_content = _decode_resource(restored_data, label=spec.filename)
        restored_errors, _ = _text_errors(
            restored_content, max_bytes=MAX_DOCUMENT_BYTES
        )
        if restored_errors:
            raise ManagementValidationError(
                "The selected backup is not a valid assistant document.",
                details={"errors": restored_errors},
            )
        backup_created = (
            _backup_resource(home, f"document-{spec.id}", data) if exists else False
        )
        path = _resource_path(home, spec.relative_path, create_parent=True)
        _require_resource_unchanged(
            path,
            expected_revision=current_revision,
            max_bytes=MAX_DOCUMENT_BYTES,
        )
        _atomic_write(path, restored_data, create_mode=spec.create_mode)
        response = _document_response(
            home,
            spec,
            exists=True,
            data=restored_data,
            content=restored_content,
            include_content=True,
        )
        response.update(
            {
                "reverted": True,
                "restoredRevision": restored_revision,
                "backupCreated": backup_created,
            }
        )
        return response


__all__ = [
    "ASSISTANT_DOCUMENTS",
    "MAX_CONFIGURATION_BYTES",
    "MAX_DOCUMENT_BYTES",
    "ManagementConflictError",
    "ManagementNotFoundError",
    "ManagementProfileError",
    "ManagementValidationError",
    "REDACTED_VALUE",
    "get_assistant_document",
    "get_management_profile_capabilities",
    "get_profile_configuration",
    "list_assistant_documents",
    "revert_assistant_document",
    "update_assistant_document",
    "update_profile_configuration",
    "validate_assistant_document",
    "validate_profile_configuration",
]

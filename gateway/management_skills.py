"""Safe, profile-scoped skills reads for the management API.

This module deliberately exposes a much narrower surface than ``skill_view``:
clients may list installed skills and read only each skill's canonical
``SKILL.md`` document.  Supporting files, installation, editing, dependency
management, and marketplace operations are outside this API boundary.

The active profile is supplied by the gateway's existing HERMES_HOME context
scope.  No profile path is accepted from an API client here.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS,
    ORG_MIRROR_DIR_NAME,
    SKILL_SUPPORT_DIRS,
    get_all_skills_dirs,
    parse_frontmatter,
    read_active_org_id,
    skill_matches_environment,
    skill_matches_platform,
)
from hermes_constants import get_hermes_home


MAX_SKILL_DOCUMENT_BYTES = 512 * 1024
MAX_SKILL_ID_LENGTH = 128
MAX_SKILL_QUERY_LENGTH = 256

_SKILL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TEXT_CONTROLS = frozenset(("\t", "\n", "\r"))


class ManagementSkillsError(Exception):
    """A normalized error suitable for the management HTTP adapter."""

    def __init__(self, code: str, message: str, *, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        # ``status_code`` is convenient for adapters that use the common
        # HTTP-exception spelling.
        self.status_code = status

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


@dataclass(frozen=True)
class _SkillRecord:
    skill_id: str
    name: str
    description: str
    source: str
    status: str
    path: Path
    root: Path
    revision: str | None
    metadata: dict[str, Any]
    read_error: ManagementSkillsError | None = None


def get_management_skills_capability() -> dict[str, bool]:
    """Return the flat capability flags advertised by the management API.

    Hermes has stable profile-aware discovery and canonical document reads.
    It does not currently expose an optimistic-concurrency skill toggle API,
    so management clients must accurately remain read-only.
    """

    return {
        "skillsList": True,
        "skillsRead": True,
        "skillsControl": False,
    }


def list_management_skills(
    query: str | None = None,
    *,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """List installed skills for the profile in the current runtime scope."""

    normalized_query = _normalize_query(query)
    records = _discover_skill_records()
    if normalized_query:
        records = [
            record
            for record in records
            if normalized_query in _search_text(record).casefold()
        ]

    return {
        "profileId": _profile_id(profile_id),
        "skills": [_summary(record) for record in records],
    }


def read_management_skill(
    skill_id: str,
    *,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Read one installed skill's canonical ``SKILL.md`` as UTF-8 text."""

    normalized_id = _normalize_skill_id(skill_id)
    record = next(
        (item for item in _discover_skill_records() if item.skill_id == normalized_id),
        None,
    )
    if record is None:
        raise ManagementSkillsError(
            "skill_not_found",
            f"Skill '{normalized_id}' was not found in this profile.",
            status=404,
        )
    if record.read_error is not None:
        raise record.read_error

    content, raw = _read_skill_document(record.path, record.root)
    revision = hashlib.sha256(raw).hexdigest()
    summary = _summary(record)
    summary["revision"] = revision
    return {
        "profileId": _profile_id(profile_id),
        "skill": {
            **summary,
            "content": content,
            "mediaType": "text/markdown",
        },
    }


def _profile_id(explicit: str | None) -> str:
    if explicit is not None:
        candidate = str(explicit).strip()
        if candidate:
            return candidate

    home = get_hermes_home()
    if home.parent.name == "profiles" and home.name:
        return home.name
    return "default"


def _normalize_query(query: str | None) -> str:
    if query is None:
        return ""
    value = str(query).strip()
    if len(value) > MAX_SKILL_QUERY_LENGTH:
        raise ManagementSkillsError(
            "invalid_query",
            f"Skill search queries are limited to {MAX_SKILL_QUERY_LENGTH} characters.",
            status=400,
        )
    if any(ord(char) < 32 and char not in _TEXT_CONTROLS for char in value):
        raise ManagementSkillsError(
            "invalid_query",
            "Skill search query contains control characters.",
            status=400,
        )
    return value.casefold()


def _normalize_skill_id(skill_id: str) -> str:
    value = str(skill_id or "").strip()
    if not value or len(value) > MAX_SKILL_ID_LENGTH or not _SKILL_ID_RE.fullmatch(value):
        raise ManagementSkillsError(
            "invalid_skill_id",
            "Skill id is invalid.",
            status=400,
        )
    return value


def _discover_skill_records() -> list[_SkillRecord]:
    disabled = _globally_disabled_skills()
    records: list[_SkillRecord] = []
    seen: set[str] = set()

    for path, root, default_source in _iter_skill_documents():
        fallback_name = path.parent.name
        if not _is_valid_skill_id(fallback_name):
            continue

        try:
            content, raw = _read_skill_document(path, root)
            frontmatter, body = parse_frontmatter(content)
            declared_name = str(frontmatter.get("name") or fallback_name).strip()
            skill_id = declared_name if _is_valid_skill_id(declared_name) else fallback_name
            description = _description(frontmatter, body)
            source = _source_for(skill_id, path, root, default_source)
            status = _status_for(skill_id, frontmatter, disabled)
            revision = hashlib.sha256(raw).hexdigest()
            metadata = _metadata(frontmatter, path, root)
            read_error = None
        except ManagementSkillsError as exc:
            skill_id = fallback_name
            description = ""
            source = _source_for(skill_id, path, root, default_source)
            status = "unavailable"
            revision = None
            metadata = {"availabilityReason": exc.code}
            read_error = exc

        # Match Hermes' existing discovery precedence: the profile skills tree
        # wins over configured shared/external roots on name collisions.
        if skill_id in seen:
            continue
        seen.add(skill_id)
        records.append(
            _SkillRecord(
                skill_id=skill_id,
                name=skill_id,
                description=description,
                source=source,
                status=status,
                path=path,
                root=root,
                revision=revision,
                metadata=metadata,
                read_error=read_error,
            )
        )

    return sorted(records, key=lambda record: (record.name.casefold(), record.skill_id))


def _iter_skill_documents() -> Iterator[tuple[Path, Path, str]]:
    """Yield canonical documents without following skill-tree symlinks.

    ``agent.skill_utils.iter_skill_index_files`` intentionally follows links so
    locally linked development skills work in the agent.  A remote management
    read boundary has a stricter threat model, so this walker reuses Hermes'
    exclusion/org rules but refuses linked directories and linked SKILL.md
    files.  Explicitly configured external roots are resolved once and treated
    as shared allowlist roots.
    """

    roots = get_all_skills_dirs()
    for index, configured_root in enumerate(roots):
        # A profile's own skills tree is the isolation boundary. Do not let a
        # linked ``skills/`` directory silently turn one profile's management
        # API into a view of another profile. Configured external roots are a
        # separate, explicit shared allowlist and may themselves be links.
        try:
            if index == 0 and configured_root.is_symlink():
                continue
        except OSError:
            continue
        try:
            root = configured_root.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not root.is_dir():
            continue

        active_org = read_active_org_id(root)
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            try:
                relative = current_path.relative_to(root)
            except ValueError:
                dirs[:] = []
                continue

            if not relative.parts:
                if active_org is None and ORG_MIRROR_DIR_NAME in dirs:
                    dirs.remove(ORG_MIRROR_DIR_NAME)
            elif relative.parts == (ORG_MIRROR_DIR_NAME,):
                dirs[:] = [name for name in dirs if name == active_org]

            has_skill_md = "SKILL.md" in files
            safe_dirs: list[str] = []
            for name in dirs:
                child = current_path / name
                if name in EXCLUDED_SKILL_DIRS:
                    continue
                if has_skill_md and name in SKILL_SUPPORT_DIRS:
                    continue
                try:
                    if child.is_symlink():
                        continue
                except OSError:
                    continue
                safe_dirs.append(name)
            dirs[:] = safe_dirs

            if "SKILL.md" not in files:
                continue
            path = current_path / "SKILL.md"
            source = "profile" if index == 0 else "shared"
            yield path, root, source


def _read_skill_document(path: Path, root: Path) -> tuple[str, bytes]:
    _validate_document_path(path, root)

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManagementSkillsError(
            "skill_document_unreadable",
            "The skill document could not be read safely.",
            status=409,
        ) from exc

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ManagementSkillsError(
                "unsafe_skill_path",
                "The canonical skill document is not a regular file.",
                status=422,
            )
        if opened.st_size > MAX_SKILL_DOCUMENT_BYTES:
            raise ManagementSkillsError(
                "skill_document_too_large",
                f"Skill documents are limited to {MAX_SKILL_DOCUMENT_BYTES} bytes.",
                status=413,
            )

        # On procfs-capable hosts, verify the file descriptor itself still
        # resolves inside the allowlisted root. This closes a parent-directory
        # swap race between lexical validation and open().
        proc_fd = Path(f"/proc/self/fd/{descriptor}")
        if proc_fd.exists():
            try:
                proc_fd.resolve(strict=True).relative_to(root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ManagementSkillsError(
                    "unsafe_skill_path",
                    "The canonical skill document escapes its skills root.",
                    status=422,
                ) from exc

        chunks: list[bytes] = []
        remaining = MAX_SKILL_DOCUMENT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_SKILL_DOCUMENT_BYTES:
            raise ManagementSkillsError(
                "skill_document_too_large",
                f"Skill documents are limited to {MAX_SKILL_DOCUMENT_BYTES} bytes.",
                status=413,
            )
    finally:
        os.close(descriptor)

    try:
        content = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ManagementSkillsError(
            "skill_document_not_text",
            "The canonical skill document is not valid UTF-8 text.",
            status=415,
        ) from exc
    if any(
        (ord(char) < 32 and char not in _TEXT_CONTROLS) or ord(char) == 127
        for char in content
    ):
        raise ManagementSkillsError(
            "skill_document_not_text",
            "The canonical skill document contains binary control characters.",
            status=415,
        )
    return content, raw


def _validate_document_path(path: Path, root: Path) -> None:
    if path.name != "SKILL.md":
        raise ManagementSkillsError(
            "unsafe_skill_path",
            "Only a skill's canonical SKILL.md document can be read.",
            status=400,
        )

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ManagementSkillsError(
            "unsafe_skill_path",
            "The canonical skill document escapes its skills root.",
            status=422,
        ) from exc

    current = root
    for part in relative.parts:
        current = current / part
        try:
            file_stat = current.lstat()
        except OSError as exc:
            raise ManagementSkillsError(
                "skill_document_unreadable",
                "The canonical skill document is unavailable.",
                status=409,
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode):
            raise ManagementSkillsError(
                "unsafe_skill_path",
                "Symlinked skill documents are not exposed by the management API.",
                status=422,
            )

    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ManagementSkillsError(
            "unsafe_skill_path",
            "The canonical skill document escapes its skills root.",
            status=422,
        ) from exc


def _globally_disabled_skills() -> set[str]:
    try:
        from hermes_cli.config import read_raw_config
        from hermes_cli.skills_config import get_disabled_skills

        return get_disabled_skills(read_raw_config())
    except Exception:
        return set()


def _source_for(skill_id: str, path: Path, root: Path, default_source: str) -> str:
    try:
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ORG_MIRROR_DIR_NAME:
            return "shared"
    except ValueError:
        return "shared"

    if default_source == "profile":
        try:
            from tools.skill_usage import is_bundled

            if is_bundled(skill_id):
                return "builtin"
        except Exception:
            pass
    return default_source


def _status_for(
    skill_id: str,
    frontmatter: Mapping[str, Any],
    disabled: set[str],
) -> str:
    if not skill_matches_platform(dict(frontmatter)):
        return "unavailable"
    if not skill_matches_environment(dict(frontmatter)):
        return "unavailable"
    if skill_id in disabled:
        return "disabled"
    return "enabled"


def _metadata(frontmatter: Mapping[str, Any], path: Path, root: Path) -> dict[str, Any]:
    nested = frontmatter.get("metadata")
    nested = nested if isinstance(nested, Mapping) else {}
    hermes = nested.get("hermes")
    hermes = hermes if isinstance(hermes, Mapping) else {}

    category = frontmatter.get("category") or hermes.get("category")
    if not category:
        try:
            relative = path.relative_to(root)
            parts = relative.parts[:-2] if len(relative.parts) >= 3 else ()
            category = parts[-1] if parts else None
        except ValueError:
            category = None

    metadata: dict[str, Any] = {}
    for key in ("version", "author", "license"):
        value = frontmatter.get(key)
        if value is not None:
            metadata[key] = _bounded_string(value, 256)
    if category is not None:
        metadata["category"] = _bounded_string(category, 128)

    tags = frontmatter.get("tags") or hermes.get("tags")
    platforms = frontmatter.get("platforms")
    environments = frontmatter.get("environments")
    if tags:
        metadata["tags"] = _string_list(tags)
    if platforms:
        metadata["platforms"] = _string_list(platforms)
    if environments:
        metadata["environments"] = _string_list(environments)
    return metadata


def _description(frontmatter: Mapping[str, Any], body: str) -> str:
    raw = frontmatter.get("description")
    if raw is not None:
        return _bounded_string(raw, 1024)
    for line in body.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            return candidate[:1024]
    return ""


def _summary(record: _SkillRecord) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": record.skill_id,
        "name": record.name,
        "description": record.description,
        "source": record.source,
        "status": record.status,
        "enabled": record.status == "enabled",
        "controllable": False,
    }
    if record.revision is not None:
        result["revision"] = record.revision
    if record.metadata:
        result["metadata"] = dict(record.metadata)
    return result


def _search_text(record: _SkillRecord) -> str:
    values = [record.name, record.description, record.source, record.status]
    for value in record.metadata.values():
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return "\n".join(values)


def _is_valid_skill_id(value: str) -> bool:
    return bool(value and len(value) <= MAX_SKILL_ID_LENGTH and _SKILL_ID_RE.fullmatch(value))


def _bounded_string(value: Any, limit: int) -> str:
    return str(value).strip()[:limit]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [item.strip() for item in value.strip("[]").split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item).strip() for item in value]
    else:
        values = [str(value).strip()]
    return [item[:128] for item in values if item][:64]

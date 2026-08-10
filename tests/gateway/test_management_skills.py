"""Behavior and security contracts for the management skills read API."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from gateway.management_skills import (
    MAX_SKILL_DOCUMENT_BYTES,
    ManagementSkillsError,
    get_management_skills_capability,
    list_management_skills,
    read_management_skill,
)


@pytest.fixture
def profile_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes" / "profiles" / "writer"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "A useful skill.",
    extra_frontmatter: str = "",
    body: str = "# Instructions\n\nDo the useful thing.\n",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    document = skill_dir / "SKILL.md"
    document.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{extra_frontmatter}"
        "---\n\n"
        f"{body}",
        encoding="utf-8",
    )
    return document


def test_capability_is_honestly_read_only() -> None:
    assert get_management_skills_capability() == {
        "skillsList": True,
        "skillsRead": True,
        "skillsControl": False,
    }


def test_list_is_profile_scoped_searchable_and_reports_metadata(
    profile_home: Path,
) -> None:
    skills = profile_home / "skills"
    platform_tag = (
        "windows"
        if sys.platform.startswith("win")
        else "macos"
        if sys.platform.startswith("darwin")
        else "linux"
    )
    _write_skill(
        skills,
        "writer",
        description="Draft release notes.",
        extra_frontmatter=(
            "version: 2.1\n"
            "author: Hermes Agent\n"
            "license: MIT\n"
            f"platforms: [{platform_tag}]\n"
            "metadata:\n"
            "  hermes:\n"
            "    category: communication\n"
            "    tags: [writing, release]\n"
        ),
    )
    _write_skill(skills, "calendar", description="Plan a calendar.")
    (skills / ".bundled_manifest").write_text("writer:deadbeef\n", encoding="utf-8")
    (profile_home / "config.yaml").write_text(
        "skills:\n  disabled:\n    - calendar\n",
        encoding="utf-8",
    )

    result = list_management_skills("release", profile_id="writer")

    assert result["profileId"] == "writer"
    assert len(result["skills"]) == 1
    [skill] = result["skills"]
    assert skill == {
        "id": "writer",
        "name": "writer",
        "description": "Draft release notes.",
        "source": "builtin",
        "status": "enabled",
        "enabled": True,
        "controllable": False,
        "revision": skill["revision"],
        "metadata": {
            "version": "2.1",
            "author": "Hermes Agent",
            "license": "MIT",
            "category": "communication",
            "tags": ["writing", "release"],
            "platforms": [platform_tag],
        },
    }
    assert len(skill["revision"]) == 64

    all_skills = {item["id"]: item for item in list_management_skills()["skills"]}
    assert all_skills["calendar"]["status"] == "disabled"
    assert all_skills["calendar"]["enabled"] is False


def test_configured_external_skills_are_shared_and_profile_skill_wins(
    profile_home: Path,
    tmp_path: Path,
) -> None:
    external = tmp_path / "team-skills"
    external.mkdir()
    _write_skill(external, "shared-one", description="Shared team workflow.")
    _write_skill(external, "duplicate", description="External copy.")
    _write_skill(profile_home / "skills", "duplicate", description="Profile copy.")
    (profile_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n",
        encoding="utf-8",
    )

    skills = {item["id"]: item for item in list_management_skills()["skills"]}

    assert skills["shared-one"]["source"] == "shared"
    assert skills["duplicate"]["source"] == "profile"
    assert skills["duplicate"]["description"] == "Profile copy."


def test_read_returns_only_canonical_markdown_with_revision(profile_home: Path) -> None:
    document = _write_skill(profile_home / "skills", "reader", body="Read this.\n")
    raw = document.read_bytes()
    support = document.parent / "references" / "private.md"
    support.parent.mkdir()
    support.write_text("not exposed", encoding="utf-8")

    result = read_management_skill("reader")

    assert result["profileId"] == "writer"
    skill = result["skill"]
    assert skill["mediaType"] == "text/markdown"
    assert skill["content"] == raw.decode("utf-8")
    assert skill["revision"] == hashlib.sha256(raw).hexdigest()
    assert "linkedFiles" not in skill
    assert "path" not in str(result).lower()


@pytest.mark.parametrize("skill_id", ["", "../SOUL.md", "a/b", "nul\0byte"])
def test_read_rejects_path_like_skill_ids(profile_home: Path, skill_id: str) -> None:
    with pytest.raises(ManagementSkillsError) as raised:
        read_management_skill(skill_id)
    assert raised.value.code == "invalid_skill_id"
    assert raised.value.status == 400


def test_read_rejects_symlinked_document(profile_home: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    skill_dir = profile_home / "skills" / "linked-file"
    skill_dir.mkdir()
    os.symlink(outside, skill_dir / "SKILL.md")

    listed = {item["id"]: item for item in list_management_skills()["skills"]}
    assert listed["linked-file"]["status"] == "unavailable"
    assert listed["linked-file"]["metadata"]["availabilityReason"] == "unsafe_skill_path"

    with pytest.raises(ManagementSkillsError) as raised:
        read_management_skill("linked-file")
    assert raised.value.code == "unsafe_skill_path"


def test_symlinked_skill_directory_is_not_traversed(profile_home: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside-skill"
    _write_skill(outside.parent, outside.name)
    os.symlink(outside, profile_home / "skills" / "linked-directory")

    assert list_management_skills()["skills"] == []
    with pytest.raises(ManagementSkillsError) as raised:
        read_management_skill("linked-directory")
    assert raised.value.code == "skill_not_found"


def test_symlinked_profile_skills_root_cannot_cross_profile_boundary(
    profile_home: Path,
    tmp_path: Path,
) -> None:
    other_profile_skills = tmp_path / "other-profile" / "skills"
    other_profile_skills.mkdir(parents=True)
    _write_skill(other_profile_skills, "other-profile-secret")
    profile_skills = profile_home / "skills"
    profile_skills.rmdir()
    os.symlink(other_profile_skills, profile_skills)

    assert list_management_skills()["skills"] == []
    with pytest.raises(ManagementSkillsError) as raised:
        read_management_skill("other-profile-secret")
    assert raised.value.code == "skill_not_found"


def test_oversized_and_non_utf8_documents_are_unavailable(profile_home: Path) -> None:
    oversized_dir = profile_home / "skills" / "oversized"
    oversized_dir.mkdir()
    (oversized_dir / "SKILL.md").write_bytes(b"x" * (MAX_SKILL_DOCUMENT_BYTES + 1))
    binary_dir = profile_home / "skills" / "binary"
    binary_dir.mkdir()
    (binary_dir / "SKILL.md").write_bytes(b"---\nname: binary\n---\n\xff\xfe")

    listed = {item["id"]: item for item in list_management_skills()["skills"]}
    assert listed["oversized"]["status"] == "unavailable"
    assert listed["binary"]["status"] == "unavailable"

    with pytest.raises(ManagementSkillsError) as oversized:
        read_management_skill("oversized")
    assert oversized.value.code == "skill_document_too_large"
    assert oversized.value.status == 413

    with pytest.raises(ManagementSkillsError) as binary:
        read_management_skill("binary")
    assert binary.value.code == "skill_document_not_text"
    assert binary.value.status == 415


def test_invalid_search_is_rejected(profile_home: Path) -> None:
    with pytest.raises(ManagementSkillsError) as raised:
        list_management_skills("x" * 257)
    assert raised.value.code == "invalid_query"

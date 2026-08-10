from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from gateway import management_profile as management


@pytest.fixture
def profile_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "selected-profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _field(response: dict, key: str) -> dict:
    return next(item for item in response["fields"] if item["key"] == key)


def test_configuration_is_profile_scoped_redacted_and_schema_derived(
    profile_home: Path,
) -> None:
    (profile_home / "config.yaml").write_text(
        """agent:
  max_turns: 42
providers:
  local:
    api_key: super-secret
    headers:
      Authorization: bearer-secret
display:
  compact: true
platforms:
  api_server:
    extra:
      key: caller-secret
""",
        encoding="utf-8",
    )

    response = management.get_profile_configuration()

    assert response["exists"] is True
    assert response["revision"].startswith("sha256:")
    assert "super-secret" not in response["content"]
    assert "bearer-secret" not in response["content"]
    assert "caller-secret" not in response["content"]
    assert management.REDACTED_VALUE in response["content"]
    assert response["redactedPaths"] == [
        "platforms.api_server.extra.key",
        "providers.local.api_key",
        "providers.local.headers",
    ]
    assert _field(response, "agent.max_turns") == {
        "key": "agent.max_turns",
        "type": "integer",
        "defaultValue": management.DEFAULT_CONFIG["agent"]["max_turns"],
        "value": 42,
        "source": "override",
        "category": "agent",
        "writable": True,
    }
    assert _field(response, "display.streaming")["source"] == "default"
    assert _field(response, "display.streaming")["value"] is False


def test_configuration_roundtrip_preserves_placeholder_and_omitted_secrets(
    profile_home: Path,
) -> None:
    config_path = profile_home / "config.yaml"
    original = (
        "providers:\n"
        "  local:\n"
        "    api_key: super-secret\n"
        "    base_url: https://example.test/v1\n"
        "agent:\n"
        "  max_turns: 20\n"
    )
    config_path.write_text(original, encoding="utf-8")
    loaded = management.get_profile_configuration()

    placeholder_draft = yaml.safe_load(loaded["content"])
    placeholder_draft["agent"]["max_turns"] = 21
    saved = management.update_profile_configuration(
        yaml.safe_dump(placeholder_draft, sort_keys=False), loaded["revision"]
    )

    authoritative = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert authoritative["providers"]["local"]["api_key"] == "super-secret"
    assert authoritative["agent"]["max_turns"] == 21
    assert management.REDACTED_VALUE not in config_path.read_text(encoding="utf-8")
    assert "super-secret" not in saved["content"]
    assert saved["saved"] is True
    assert saved["backupCreated"] is True

    omitted_draft = {"agent": {"max_turns": 22}}
    saved_again = management.update_profile_configuration(
        yaml.safe_dump(omitted_draft), saved["revision"]
    )
    authoritative = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert authoritative["providers"]["local"]["api_key"] == "super-secret"
    assert authoritative["agent"]["max_turns"] == 22
    assert any(
        warning["code"] == "secrets_preserved" for warning in saved_again["warnings"]
    )

    backups = list((profile_home / "backups" / "management" / "configuration").glob("*.bak"))
    assert len(backups) == 2
    assert any(b"super-secret" in backup.read_bytes() for backup in backups)
    if os.name == "posix":
        assert all((backup.stat().st_mode & 0o777) == 0o600 for backup in backups)


def test_configuration_rejects_secret_edits_and_secret_list_reordering(
    profile_home: Path,
) -> None:
    config_path = profile_home / "config.yaml"
    config_path.write_text(
        """custom_providers:
  - name: first
    base_url: https://first.test/v1
    api_key: first-secret
  - name: second
    base_url: https://second.test/v1
    api_key: second-secret
""",
        encoding="utf-8",
    )
    loaded = management.get_profile_configuration()
    draft = yaml.safe_load(loaded["content"])
    draft["custom_providers"][0]["api_key"] = "replacement"

    validation = management.validate_profile_configuration(
        yaml.safe_dump(draft, sort_keys=False)
    )
    assert validation["valid"] is False
    assert any(error["code"] == "secret_not_editable" for error in validation["errors"])

    draft = yaml.safe_load(loaded["content"])
    draft["custom_providers"].reverse()
    validation = management.validate_profile_configuration(
        yaml.safe_dump(draft, sort_keys=False)
    )
    assert validation["valid"] is False
    assert any(
        error["code"] == "secret_container_changed" for error in validation["errors"]
    )
    assert "first-secret" in config_path.read_text(encoding="utf-8")


def test_configuration_validation_conflict_and_precondition(
    profile_home: Path,
) -> None:
    config_path = profile_home / "config.yaml"
    config_path.write_text("agent:\n  max_turns: 10\n", encoding="utf-8")
    loaded = management.get_profile_configuration()

    invalid = management.validate_profile_configuration("agent: [unterminated")
    assert invalid["valid"] is False
    assert invalid["errors"][0]["code"] == "invalid_yaml"
    assert invalid["errors"][0]["line"] == 1

    config_path.write_text("agent:\n  max_turns: 11\n", encoding="utf-8")
    with pytest.raises(management.ManagementConflictError) as conflict:
        management.update_profile_configuration(
            "agent:\n  max_turns: 12\n", loaded["revision"]
        )
    current = management.get_profile_configuration()
    assert conflict.value.status_code == 409
    assert conflict.value.details["currentRevision"] == current["revision"]
    assert config_path.read_text(encoding="utf-8") == "agent:\n  max_turns: 11\n"

    with pytest.raises(management.ManagementProfileError) as precondition:
        management.update_profile_configuration("{}\n", "")
    assert precondition.value.code == "precondition_required"
    assert precondition.value.status_code == 428


def test_configuration_atomic_failure_keeps_authoritative_file(
    profile_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = profile_home / "config.yaml"
    original = b"agent:\n  max_turns: 10\n"
    config_path.write_bytes(original)
    loaded = management.get_profile_configuration()
    real_replace = management.os.replace

    def fail_primary_replace(source: str, target: str | os.PathLike[str]) -> None:
        if Path(target) == config_path:
            raise OSError("simulated replace failure")
        real_replace(source, target)

    monkeypatch.setattr(management.os, "replace", fail_primary_replace)
    with pytest.raises(management.ManagementProfileError) as failure:
        management.update_profile_configuration(
            "agent:\n  max_turns: 99\n", loaded["revision"]
        )
    assert failure.value.code == "resource_write_failed"
    assert config_path.read_bytes() == original


def test_managed_install_advertises_read_only_configuration_and_blocks_writes(
    profile_home: Path,
) -> None:
    config_path = profile_home / "config.yaml"
    config_path.write_text("agent:\n  max_turns: 10\n", encoding="utf-8")
    (profile_home / ".managed").write_text("nixos\n", encoding="utf-8")

    capabilities = management.get_management_profile_capabilities()
    assert capabilities == {
        "profileConfigurationRead": True,
        "profileConfigurationWrite": False,
        "assistantDocumentsRead": True,
        "assistantDocumentsWrite": True,
    }
    loaded = management.get_profile_configuration()
    assert loaded["writable"] is False
    assert all(field["writable"] is False for field in loaded["fields"])

    with pytest.raises(management.ManagementProfileError) as validation:
        management.validate_profile_configuration("agent:\n  max_turns: 11\n")
    assert validation.value.code == "managed_configuration"
    assert validation.value.status_code == 403

    with pytest.raises(management.ManagementProfileError) as update:
        management.update_profile_configuration(
            "agent:\n  max_turns: 11\n", loaded["revision"]
        )
    assert update.value.code == "managed_configuration"
    assert config_path.read_text(encoding="utf-8") == "agent:\n  max_turns: 10\n"

    soul = management.get_assistant_document("soul")
    management.update_assistant_document("soul", "# Managed soul\n", soul["revision"])
    assert (profile_home / "SOUL.md").read_text(encoding="utf-8") == "# Managed soul\n"


def test_documents_use_the_real_allowlist_and_load_content_on_demand(
    profile_home: Path,
) -> None:
    (profile_home / "SOUL.md").write_text("# Soul\n", encoding="utf-8")
    memories = profile_home / "memories"
    memories.mkdir()
    (memories / "USER.md").write_text("# User\n", encoding="utf-8")

    listing = management.list_assistant_documents()

    assert [item["id"] for item in listing["documents"]] == ["soul", "user", "memory"]
    assert [item["filename"] for item in listing["documents"]] == [
        "SOUL.md",
        "USER.md",
        "MEMORY.md",
    ]
    assert all("content" not in item for item in listing["documents"])
    missing = next(item for item in listing["documents"] if item["id"] == "memory")
    assert missing["exists"] is False
    assert missing["sizeBytes"] == 0

    soul = management.get_assistant_document("soul")
    assert soul["content"] == "# Soul\n"
    assert soul["lineEnding"] == "lf"
    assert soul["finalNewline"] is True

    with pytest.raises(management.ManagementNotFoundError) as unsupported:
        management.get_assistant_document("../.env")
    assert unsupported.value.code == "document_not_found"


def test_document_save_preserves_line_endings_and_reverts_backup(
    profile_home: Path,
) -> None:
    soul_path = profile_home / "SOUL.md"
    original = b"# Soul\r\nold"
    soul_path.write_bytes(original)
    loaded = management.get_assistant_document("soul")
    assert loaded["lineEnding"] == "crlf"
    assert loaded["finalNewline"] is False

    saved = management.update_assistant_document(
        "soul", "# Soul\nnew\n", loaded["revision"]
    )

    assert soul_path.read_bytes() == b"# Soul\r\nnew"
    assert saved["lineEnding"] == "crlf"
    assert saved["finalNewline"] is False
    assert saved["backupAvailable"] is True
    assert saved["backupRevision"] is not None

    reverted = management.revert_assistant_document(
        "soul", saved["revision"], backup_revision=saved["backupRevision"]
    )
    assert reverted["reverted"] is True
    assert reverted["restoredRevision"] == saved["backupRevision"]
    assert soul_path.read_bytes() == original
    assert reverted["content"] == "# Soul\r\nold"


def test_document_conflict_missing_backup_and_creation_modes(
    profile_home: Path,
) -> None:
    missing = management.get_assistant_document("user")
    with pytest.raises(management.ManagementNotFoundError) as no_backup:
        management.revert_assistant_document("user", missing["revision"])
    assert no_backup.value.code == "backup_not_found"

    created = management.update_assistant_document(
        "user",
        "# User\nAda\n",
        missing["revision"],
        line_ending="lf",
        final_newline=True,
    )
    user_path = profile_home / "memories" / "USER.md"
    assert user_path.read_bytes() == b"# User\nAda\n"
    if os.name == "posix":
        assert (user_path.stat().st_mode & 0o777) == 0o600

    user_path.write_text("external\n", encoding="utf-8")
    with pytest.raises(management.ManagementConflictError):
        management.update_assistant_document("user", "draft\n", created["revision"])
    assert user_path.read_text(encoding="utf-8") == "external\n"


def test_document_validation_enforces_text_and_size_limits(profile_home: Path) -> None:
    control = management.validate_assistant_document("soul", "hello\x00world")
    assert control["valid"] is False
    assert control["errors"][0]["code"] == "invalid_text_control"

    invalid_convention = management.validate_assistant_document(
        "soul", "hello", line_ending=["lf"]  # type: ignore[arg-type]
    )
    assert invalid_convention["valid"] is False
    assert invalid_convention["errors"][0]["code"] == "invalid_line_ending"

    oversized = management.validate_assistant_document(
        "soul", "x" * (management.MAX_DOCUMENT_BYTES + 1)
    )
    assert oversized["valid"] is False
    assert oversized["errors"][0]["code"] == "resource_too_large"

    (profile_home / "SOUL.md").write_bytes(b"x" * (management.MAX_DOCUMENT_BYTES + 1))
    with pytest.raises(management.ManagementProfileError) as too_large:
        management.get_assistant_document("soul")
    assert too_large.value.code == "resource_too_large"
    assert too_large.value.status_code == 413


def test_symlinks_cannot_escape_profile_for_documents_or_backups(
    profile_home: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (profile_home / "SOUL.md").symlink_to(outside)

    with pytest.raises(management.ManagementProfileError) as linked_document:
        management.get_assistant_document("soul")
    assert linked_document.value.code == "unsafe_symlink"
    assert outside.read_text(encoding="utf-8") == "outside"

    (profile_home / "SOUL.md").unlink()
    (profile_home / "SOUL.md").write_text("inside", encoding="utf-8")
    loaded = management.get_assistant_document("soul")
    outside_backups = tmp_path / "outside-backups"
    outside_backups.mkdir()
    (profile_home / "backups").symlink_to(outside_backups, target_is_directory=True)

    with pytest.raises(management.ManagementProfileError) as linked_backup:
        management.update_assistant_document("soul", "changed", loaded["revision"])
    assert linked_backup.value.code == "unsafe_symlink"
    assert (profile_home / "SOUL.md").read_text(encoding="utf-8") == "inside"
    assert list(outside_backups.iterdir()) == []

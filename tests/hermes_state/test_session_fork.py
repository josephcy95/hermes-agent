"""Behavior contracts for atomic full and point-in-time session forks."""

import json

import pytest

from hermes_state import SessionDB, SessionForkError


@pytest.fixture()
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


def _tool_calls(*call_ids: str):
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": f"tool_{index}", "arguments": "{}"},
        }
        for index, call_id in enumerate(call_ids, start=1)
    ]


def test_full_fork_preserves_parent_and_inherits_runtime_state(db):
    lock = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet",
        "model_options": {"reasoning_effort": "high"},
        "route_source": "raw_request",
        "confirmed": True,
    }
    db.create_session(
        "parent",
        source="api_server",
        model="anthropic/claude-sonnet",
        model_config={
            "browser_model_lock": lock,
            "reasoning_config": {"enabled": True, "effort": "high"},
            "_delegate_from": "orchestrator",
        },
        system_prompt="parent system prompt",
    )
    db.append_message("parent", role="user", content="question")
    db.append_message("parent", role="assistant", content="answer")

    parent_before = db.get_session("parent")
    parent_messages_before = db.get_messages("parent")
    child = db.fork_session("parent", "child", title="alternate")

    parent_after = db.get_session("parent")
    parent_messages_after = db.get_messages("parent")
    for field in (
        "ended_at",
        "end_reason",
        "model",
        "model_config",
        "system_prompt",
        "message_count",
        "tool_call_count",
    ):
        assert parent_after[field] == parent_before[field]
    assert parent_messages_after == parent_messages_before

    assert child["parent_session_id"] == "parent"
    assert child["model"] == parent_before["model"]
    assert child["system_prompt"] == "parent system prompt"
    assert child["title"] == "alternate"
    child_config = json.loads(child["model_config"])
    assert child_config["browser_model_lock"] == lock
    assert child_config["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert child_config["_branched_from"] == "parent"
    assert child_config["_runtime_inherited_from"] == "parent"
    assert "_delegate_from" not in child_config
    assert [message["content"] for message in db.get_messages("child")] == [
        "question",
        "answer",
    ]


def test_point_in_time_fork_extends_through_complete_tool_group(db):
    db.create_session("parent", source="api_server")
    user_id = db.append_message("parent", role="user", content="question")
    call_id = db.append_message(
        "parent",
        role="assistant",
        content="checking",
        tool_calls=_tool_calls("call_1", "call_2"),
        finish_reason="tool_calls",
    )
    first_result = db.append_message(
        "parent", role="tool", content="one", tool_call_id="call_1"
    )
    second_result = db.append_message(
        "parent", role="tool", content="two", tool_call_id="call_2"
    )
    db.append_message("parent", role="assistant", content="final")

    db.fork_session(
        "parent", "child", through_message_id=call_id, title="tool boundary"
    )

    child_messages = db.get_messages("child")
    assert [message["role"] for message in child_messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [message["content"] for message in child_messages] == [
        "question",
        "checking",
        "one",
        "two",
    ]
    assert user_id < call_id < first_result < second_result


def test_unsettled_foreign_and_inactive_cutoffs_create_no_child(db):
    db.create_session("parent", source="api_server")
    pending_call = db.append_message(
        "parent",
        role="assistant",
        content="pending",
        tool_calls=_tool_calls("call_pending"),
        finish_reason="tool_calls",
    )

    with pytest.raises(SessionForkError) as incomplete:
        db.fork_session("parent", "incomplete-child")
    assert incomplete.value.code == "fork_unsettled"
    assert incomplete.value.status == 409
    assert db.get_session("incomplete-child") is None

    # A cutoff before the pending group is safe, but a message from another
    # session must never be accepted as a source boundary.
    db.create_session("foreign", source="api_server")
    foreign_id = db.append_message("foreign", role="user", content="foreign")
    with pytest.raises(SessionForkError) as foreign:
        db.fork_session(
            "parent", "foreign-child", through_message_id=foreign_id
        )
    assert foreign.value.code == "message_not_in_session"
    assert db.get_session("foreign-child") is None

    db.create_session("inactive", source="api_server")
    inactive_id = db.append_message("inactive", role="user", content="rewind me")
    db.rewind_to_message("inactive", inactive_id)
    with pytest.raises(SessionForkError) as inactive:
        db.fork_session(
            "inactive", "inactive-child", through_message_id=inactive_id
        )
    assert inactive.value.code == "source_message_inactive"
    assert inactive.value.status == 409
    assert db.get_session("inactive-child") is None
    assert db.get_session("parent")["end_reason"] is None
    assert pending_call > 0


def test_relationship_types_use_markers_and_parent_state_without_title_guessing(db):
    db.create_session("root", source="api_server")
    db.create_session(
        "fork",
        source="api_server",
        parent_session_id="root",
        model_config={"_branched_from": "root"},
    )
    db.create_session(
        "delegate",
        source="tool",
        parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    db.create_session("generic", source="api_server", parent_session_id="root")

    db.create_session("compressed", source="api_server")
    db.end_session("compressed", "compression")
    db.create_session(
        "continuation",
        source="subagent",
        parent_session_id="compressed",
        # Compression continuations can inherit a marker pointing elsewhere.
        model_config={"_delegate_from": "some-original-parent"},
    )

    relationships = db.get_session_relationship_types(
        ["root", "fork", "delegate", "generic", "continuation"]
    )
    assert relationships == {
        "root": "root",
        "fork": "fork",
        "delegate": "subagent",
        "generic": None,
        "continuation": "compression_continuation",
    }


def test_point_in_time_fork_handles_legacy_idless_assistant_tool_calls(db):
    db.create_session("legacy-parent", source="api_server")
    call_message_id = db.append_message(
        "legacy-parent",
        role="assistant",
        content="checking",
        # Older persistence paths stored no ids on the assistant row.
        tool_calls=[{"name": "search", "arguments": "{}"}],
        finish_reason="tool_calls",
    )
    db.append_message(
        "legacy-parent",
        role="tool",
        content="result",
        tool_call_id="call_legacy",
        tool_name="search",
    )

    db.fork_session(
        "legacy-parent", "legacy-child", through_message_id=call_message_id
    )
    assert [message["role"] for message in db.get_messages("legacy-child")] == [
        "assistant",
        "tool",
    ]

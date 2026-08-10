"""Durable response telemetry keyed by SessionDB message row id."""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def test_response_metrics_merge_targets_exact_assistant_row(db):
    db.create_session("s1", source="api_server")
    db.append_message("s1", role="user", content="same")
    first = db.append_message(
        "s1",
        role="assistant",
        content="same",
        display_metadata={"existing": "keep"},
    )
    second = db.append_message("s1", role="assistant", content="same")
    latest_without_metrics = db.append_message(
        "s1",
        role="assistant",
        content="latest",
        display_metadata={"other": "metadata"},
    )
    metrics = {
        "elapsed_ms": 321,
        "context_used": 1200,
        "context_max": 8000,
        "context_percent": 15.0,
    }

    assert db.merge_message_display_metadata("s1", second, {"response_metrics": metrics}) is True
    assert db.merge_message_display_metadata("s1", first, {"not_metrics": True}) is True
    assert db.merge_message_display_metadata("s1", first, {"response_metrics": metrics}) is True
    assert db.merge_message_display_metadata("other", second, {"response_metrics": metrics}) is False

    rows = db.get_messages("s1")
    first_row = next(row for row in rows if row["id"] == first)
    second_row = next(row for row in rows if row["id"] == second)
    latest_row = next(row for row in rows if row["id"] == latest_without_metrics)
    assert first_row["display_metadata"] == {
        "existing": "keep",
        "not_metrics": True,
        "response_metrics": metrics,
    }
    assert second_row["display_metadata"] == {"response_metrics": metrics}
    assert latest_row["display_metadata"] == {"other": "metadata"}
    assert db.get_latest_response_context_usage("s1") == {
        "context_used": 1200,
        "context_max": 8000,
        "context_percent": 15.0,
    }


def test_response_metrics_does_not_update_non_assistant_row(db):
    db.create_session("s1", source="api_server")
    user_id = db.append_message("s1", role="user", content="same")
    assert db.merge_message_display_metadata(
        "s1", user_id, {"response_metrics": {"elapsed_ms": 1}}
    ) is False

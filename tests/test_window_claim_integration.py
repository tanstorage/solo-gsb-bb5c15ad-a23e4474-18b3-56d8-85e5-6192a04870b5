"""The worker's atomic claim and the API's next_open_at share one judgement.

Regression coverage for the autumn fall-back repeated hour: a cross-midnight
window (America/New_York, Saturday 23:00 -> Sunday 01:30) closes at the
first real 01:30. During the second pass of 01:00-01:30 the claim gate must
stay closed, the head task must be rescheduled to the next true opening
(the following Saturday 23:00), no rate-limit token may be consumed (the
window gate runs before the token bucket), and the API projection must
report the very same next_open_at.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.db import Database
from app.delivery_window import DeliveryWindow
from app.main import _delivery_window_out

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

ZONE = "America/New_York"
WEEKDAYS = [6]          # opens Saturdays only
START = 23 * 3600       # 23:00
END = 90 * 60           # 01:30 (next day)

# Fall-back weekend 2025-11-01/02: 01:00-01:59:59 occurs twice on Sunday.
DUE_SINCE = datetime(2025, 11, 1, 0, 0, tzinfo=NY).timestamp()              # task due seed
OPEN_SAT = datetime(2025, 11, 1, 23, 30, tzinfo=NY).timestamp()             # open
FIRST_CLOSE = datetime(2025, 11, 2, 1, 30, tzinfo=NY, fold=0).timestamp()   # 05:30 UTC
SECOND_PASS = datetime(2025, 11, 2, 1, 15, tzinfo=NY, fold=1).timestamp()   # 06:15 UTC
NEXT_OPEN = datetime(2025, 11, 8, 23, 0, tzinfo=NY).timestamp()             # next Sat 23:00 EST

WINDOW = DeliveryWindow(ZONE, (6,), START, END)


@pytest.fixture()
def db(tmp_path):
    database = Database(str(tmp_path / "relay.db"))
    try:
        yield database
    finally:
        database.close()


def _make_windowed_subscription(db: Database) -> tuple[dict, dict]:
    """One subscription with the fall-back window and one already-due task."""
    sub = db.create_subscription("http://localhost:9001/hook", "secret")
    db.set_delivery_window(sub["id"], ZONE, list(WEEKDAYS), START, END)
    db.create_event("evt-fold", json.dumps({"n": 1}))
    # Seed the schedule in the (synthetic) past so the task is due at every
    # evaluated instant; the window gate alone decides claimability.
    task, created = db.create_task("evt-fold", sub["id"], 4, deliver_at=DUE_SINCE)
    assert created
    return sub, task


def _task_row(db: Database, task_id: int):
    return db._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def _rate_tokens(db: Database, sub_id: str) -> float:
    row = db._conn.execute(
        "SELECT rate_tokens FROM subscriptions WHERE id = ?", (sub_id,)
    ).fetchone()
    return row["rate_tokens"]


def test_claim_during_second_fold_pass_stays_closed_and_reschedules(db):
    sub, task = _make_windowed_subscription(db)

    # Second pass of the repeated hour: the window is closed, so the atomic
    # claim must refuse the task and push it to the next true opening --
    # not to the fold=1 01:00 happening minutes later.
    assert not WINDOW.is_open(SECOND_PASS)
    assert db.claim_due_task("worker-1", SECOND_PASS) is None

    row = _task_row(db, task["id"])
    assert row["status"] == "pending"
    assert row["next_attempt_at"] == pytest.approx(NEXT_OPEN)
    # The reschedule is the same pure judgement next_open_at exposes.
    assert row["next_attempt_at"] == pytest.approx(WINDOW.next_open(SECOND_PASS))


def test_claim_at_first_close_does_not_wait_for_second_pass(db):
    sub, task = _make_windowed_subscription(db)

    # At the first real 01:30 the window is already closed; the next opening
    # is next Saturday 23:00, never the second pass of the repeated hour.
    assert db.claim_due_task("worker-1", FIRST_CLOSE) is None

    row = _task_row(db, task["id"])
    assert row["status"] == "pending"
    assert row["next_attempt_at"] == pytest.approx(NEXT_OPEN)


def test_claim_while_window_open_succeeds(db):
    sub, task = _make_windowed_subscription(db)

    claimed = db.claim_due_task("worker-1", OPEN_SAT)
    assert claimed is not None
    assert claimed["id"] == task["id"]
    assert _task_row(db, task["id"])["status"] == "in_progress"


def test_next_open_at_api_projection_matches_claim_judgement(db):
    sub, task = _make_windowed_subscription(db)

    # The read-only API projection and the worker claim evaluate the same
    # window functions at the same instant, so they can never disagree.
    for now in (OPEN_SAT, FIRST_CLOSE, SECOND_PASS):
        out = _delivery_window_out(now, db.get_delivery_window(sub["id"]))
        assert out["is_open_now"] == WINDOW.is_open(now)
        expected_next = None if out["is_open_now"] else WINDOW.next_open(now)
        if expected_next is None:
            assert out["next_open_at"] is None
        else:
            assert out["next_open_at"] == datetime.fromtimestamp(
                expected_next, tz=UTC
            ).isoformat()

    # And the persisted reschedule produced by a closed claim is exactly the
    # next_open_at the API reports for that instant.
    assert db.claim_due_task("worker-1", SECOND_PASS) is None
    out = _delivery_window_out(SECOND_PASS, db.get_delivery_window(sub["id"]))
    assert out["is_open_now"] is False
    assert out["next_open_at"] == datetime.fromtimestamp(NEXT_OPEN, tz=UTC).isoformat()
    assert _task_row(db, task["id"])["next_attempt_at"] == pytest.approx(NEXT_OPEN)


def test_window_closed_claim_spends_no_token(db):
    sub, task = _make_windowed_subscription(db)
    db.set_rate_policy(sub["id"], 1.0, 1.0)  # burst 1, 1 token/second

    # Window closed (second fold pass): the claim is refused by the window
    # gate, which runs before the token bucket, so no token is spent and the
    # reschedule targets the next window opening, not a token-refill time.
    assert db.claim_due_task("worker-1", SECOND_PASS) is None
    assert _rate_tokens(db, sub["id"]) == pytest.approx(1.0)
    assert _task_row(db, task["id"])["next_attempt_at"] == pytest.approx(NEXT_OPEN)


def test_window_open_claim_spends_token(db):
    sub, task = _make_windowed_subscription(db)
    db.set_rate_policy(sub["id"], 1.0, 1.0)

    # Window open: the claim passes the window gate and then spends a token.
    claimed = db.claim_due_task("worker-1", OPEN_SAT)
    assert claimed is not None
    assert claimed["id"] == task["id"]
    assert _rate_tokens(db, sub["id"]) == pytest.approx(0.0)

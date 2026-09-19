"""Regression tests for delivery-window DST boundary handling.

Focus: the autumn fall-back repeated hour for cross-midnight windows. A
window such as America/New_York Saturday 23:00 -> Sunday 01:30 is bound to
one explicit UTC interval: it opens at Saturday's 23:00 and closes at the
FIRST real occurrence of Sunday 01:30 (fold=0); the second pass of
01:00-01:30 (fold=1) must neither re-open nor extend it. Both folds are
exercised, alongside spring-forward gaps, weekday attribution and the
next_open judgement the worker claim reuses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.delivery_window import DeliveryWindow

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

# 2025-11-02: US fall-back -- 02:00 EDT falls back to 01:00 EST, so the wall
# times 01:00..01:59:59 occur twice (fold=0 in EDT, fold=1 in EST).
# 2025-03-09: US spring-forward -- 02:00 EST jumps straight to 03:00 EDT.


def ts(y, mo, d, h, mi, s=0, fold=0, zone=NY) -> float:
    """Epoch seconds of a local wall time; fold selects the repeated-hour pass."""
    return datetime(y, mo, d, h, mi, s, tzinfo=zone, fold=fold).timestamp()


# The window under test: opens Saturdays 23:00, closes Sundays 01:30.
WRAP = DeliveryWindow("America/New_York", (6,), 23 * 3600, 90 * 60)

# Absolute UTC bounds of the fall-back weekend's single open interval:
# Saturday 23:00 EDT = 03:00 UTC, first Sunday 01:30 (EDT) = 05:30 UTC.
OPEN_UTC = ts(2025, 11, 1, 23, 0)
FIRST_CLOSE_UTC = ts(2025, 11, 2, 1, 30, fold=0)
NEXT_SAT_OPEN = ts(2025, 11, 8, 23, 0)  # 2025-11-09 04:00 UTC (EST)


def test_open_interval_is_bound_to_explicit_utc_instants():
    assert datetime.fromtimestamp(OPEN_UTC, UTC).isoformat() == "2025-11-02T03:00:00+00:00"
    assert datetime.fromtimestamp(FIRST_CLOSE_UTC, UTC).isoformat() == "2025-11-02T05:30:00+00:00"
    # The second pass of the end wall time is a full hour later in UTC and
    # must be irrelevant to the window.
    second_close = ts(2025, 11, 2, 1, 30, fold=1)
    assert datetime.fromtimestamp(second_close, UTC).isoformat() == "2025-11-02T06:30:00+00:00"
    # Sweep the whole transition weekend in 5-minute steps: the window is
    # open exactly on [OPEN_UTC, FIRST_CLOSE_UTC) and nowhere else.
    start = datetime(2025, 11, 1, 0, 0, tzinfo=UTC).timestamp()
    cur = start
    while cur < start + 2 * 24 * 3600:
        assert WRAP.is_open(cur) == (OPEN_UTC <= cur < FIRST_CLOSE_UTC), (
            f"is_open mismatch at {datetime.fromtimestamp(cur, UTC).isoformat()}"
        )
        cur += 300


def test_fallback_wrap_window_closes_at_first_end_occurrence():
    assert not WRAP.is_open(ts(2025, 11, 1, 22, 59, 59))
    assert WRAP.is_open(ts(2025, 11, 1, 23, 0))  # Saturday opening
    assert WRAP.is_open(ts(2025, 11, 2, 0, 30))  # post-midnight tail
    # First (fold=0) pass of the repeated hour: open until 01:30 exactly.
    assert WRAP.is_open(ts(2025, 11, 2, 1, 0, fold=0))
    assert WRAP.is_open(ts(2025, 11, 2, 1, 29, 59, fold=0))
    assert not WRAP.is_open(ts(2025, 11, 2, 1, 30, fold=0))


def test_fallback_second_pass_does_not_reopen_or_extend():
    # The whole second (fold=1) pass of 01:00-01:30 stays closed.
    assert not WRAP.is_open(ts(2025, 11, 2, 1, 0, fold=1))
    assert not WRAP.is_open(ts(2025, 11, 2, 1, 15, fold=1))
    assert not WRAP.is_open(ts(2025, 11, 2, 1, 29, 59, fold=1))
    assert not WRAP.is_open(ts(2025, 11, 2, 1, 30, fold=1))
    assert not WRAP.is_open(ts(2025, 11, 2, 2, 0))


def test_next_open_skips_second_pass_and_lands_on_next_week():
    # Evaluated at the first-pass close and anywhere inside the second pass,
    # the next opening is next Saturday 23:00 -- never the fold=1 01:00.
    assert WRAP.next_open(ts(2025, 11, 2, 1, 30, fold=0)) == NEXT_SAT_OPEN
    assert WRAP.next_open(ts(2025, 11, 2, 1, 0, fold=1)) == NEXT_SAT_OPEN
    assert WRAP.next_open(ts(2025, 11, 2, 1, 29, 59, fold=1)) == NEXT_SAT_OPEN
    assert WRAP.next_open(ts(2025, 11, 2, 2, 0)) == NEXT_SAT_OPEN
    # Before the window ever opened, the next opening is the same Saturday.
    assert WRAP.next_open(ts(2025, 11, 1, 22, 0)) == OPEN_UTC


def test_fallback_opening_inside_repeated_hour_uses_first_fold():
    # Non-wrap Sunday window 01:30-03:00: opens at the first real 01:30 and
    # stays continuously open through the repeated hour (no fold=1 hole).
    w = DeliveryWindow("America/New_York", (7,), 90 * 60, 3 * 3600)
    assert not w.is_open(ts(2025, 11, 2, 1, 29, 59, fold=0))
    assert w.is_open(ts(2025, 11, 2, 1, 30, fold=0))
    assert w.is_open(ts(2025, 11, 2, 1, 0, fold=1))
    assert w.is_open(ts(2025, 11, 2, 1, 29, 59, fold=1))
    assert w.is_open(ts(2025, 11, 2, 2, 59, 59))
    assert not w.is_open(ts(2025, 11, 2, 3, 0))


def test_spring_forward_gap_opening_resolves_to_first_real_instant():
    # 2025-03-09: 02:30 does not exist (02:00 EST -> 03:00 EDT).
    w = DeliveryWindow("America/New_York", (7,), 2 * 3600 + 1800, 4 * 3600)
    assert not w.is_open(ts(2025, 3, 9, 1, 59, 59))
    assert w.is_open(ts(2025, 3, 9, 3, 0))  # first real instant >= 02:30
    assert w.is_open(ts(2025, 3, 9, 3, 59, 59))
    assert not w.is_open(ts(2025, 3, 9, 4, 0))


def test_spring_forward_wrap_window_end_in_gap_closes_at_first_real_instant():
    # Saturday 23:00 -> Sunday 02:30; the wall clock jumps over 02:30, so
    # the window closes at 03:00 EDT (the first real instant >= 02:30).
    w = DeliveryWindow("America/New_York", (6,), 23 * 3600, 2 * 3600 + 1800)
    assert w.is_open(ts(2025, 3, 8, 23, 0))
    assert w.is_open(ts(2025, 3, 9, 1, 59, 59))
    assert not w.is_open(ts(2025, 3, 9, 3, 0))


def test_weekday_attribution_across_midnight():
    # The tail belongs to Saturday's segment: open early Sunday, but the
    # window does not open again on Sunday evening (Sunday is not an
    # opening weekday) and leaves no tail on Monday.
    assert WRAP.is_open(ts(2025, 11, 2, 0, 30))
    assert not WRAP.is_open(ts(2025, 11, 2, 23, 30))
    assert not WRAP.is_open(ts(2025, 11, 3, 0, 30))
    assert not WRAP.is_open(ts(2025, 10, 31, 23, 30))  # Friday night


def test_wrap_window_on_regular_weekend_without_dst():
    # One week earlier (2025-10-25/26) has no transition: plain 2.5h window.
    assert not WRAP.is_open(ts(2025, 10, 25, 22, 59, 59))
    assert WRAP.is_open(ts(2025, 10, 25, 23, 0))
    assert WRAP.is_open(ts(2025, 10, 26, 1, 29, 59))
    assert not WRAP.is_open(ts(2025, 10, 26, 1, 30))
    assert WRAP.next_open(ts(2025, 10, 26, 2, 0)) == ts(2025, 11, 1, 23, 0)


def test_midnight_to_midnight_window_stays_open_all_day():
    w = DeliveryWindow("America/New_York", (3,), 0, 0)  # Wednesday, 00:00-00:00
    assert w.wraps_past_midnight
    assert w.is_open(ts(2025, 11, 5, 0, 0))
    assert w.is_open(ts(2025, 11, 5, 23, 59, 59))
    assert not w.is_open(ts(2025, 11, 6, 0, 0))

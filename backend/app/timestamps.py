"""
How this application represents time. One convention, in one place.

THE RULE: every timestamp is stored as a NAIVE datetime in UTC, and is sent to
clients with an explicit `Z` suffix saying so.

WHY THIS MODULE EXISTS
----------------------
There used to be two clocks, and rows carried both.

  - Columns defaulted with `server_default=func.now()` are stamped by SQLite's
    CURRENT_TIMESTAMP, which is **UTC**.
  - Columns assigned in Python with `datetime.now()` get the machine's **local**
    time.

Both are naive — neither records which one it is — so nothing downstream could
tell them apart. One training_jobs row held `created_at 07:13:20` (UTC) beside
`started_at 12:43:20` (local, UTC+05:30): the same instant, written five and a
half hours apart, implying a job that waited half a working day to start.

The second half of the bug was on the wire. A naive datetime serialises as
"2026-07-20T07:13:20" with no zone, and `new Date(...)` in JavaScript reads a
string in that form as LOCAL time. So a UTC value was re-interpreted as local
and every relative time on the Projects page was wrong by the machine's offset —
work done a minute ago read as "6 hours ago".

WHY UTC RATHER THAN LOCAL
-------------------------
Local time is not monotonic: it jumps at a DST boundary and changes when the
machine travels, so durations computed across one are wrong and ordering can
invert. UTC has neither problem. It also matches what the majority of columns
already did, so unifying on it left the bulk of existing data correct.

Storing NAIVE UTC (rather than timezone-aware) keeps the column type unchanged
and matches what SQLite's CURRENT_TIMESTAMP already writes, so no existing row
had to be rewritten to adopt the convention — only the local-time ones, which
`scripts/backfill_utc_timestamps.py` handles.

Display stays local: the frontend renders with `toLocaleString()` /
`Intl.RelativeTimeFormat`, which convert correctly the moment the value on the
wire is honestly marked as UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import PlainSerializer


def utcnow() -> datetime:
    """The current time as a NAIVE datetime in UTC.

    Use this instead of `datetime.now()` for anything that gets stored.
    `datetime.now()` returns local time, which is the bug this module exists to
    prevent; `datetime.utcnow()` is deprecated in 3.12+.

    Naive rather than aware so it drops into the existing `DateTime` columns
    unchanged and sorts against values written by `func.now()`.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_utc_iso(value: datetime | None) -> str | None:
    """Serialise a stored timestamp as ISO-8601 explicitly marked UTC.

    This is the half that makes clients correct. Without the marker a naive
    string is read as local time by JavaScript, browsers, and most parsers —
    silently, and only wrong by the reader's offset, which is exactly the kind
    of error that survives review because it looks plausible.

    A value that already carries a zone is converted rather than relabelled, so
    this stays correct if a column ever becomes timezone-aware.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


#: The type to use for every datetime field on a RESPONSE schema.
#:
#: Annotating the field rather than adding a serialiser per schema means the
#: convention travels with the type: a new endpoint that writes `UtcDatetime`
#: is correct by construction, and one that writes a bare `datetime` is visibly
#: different from its neighbours.
UtcDatetime = Annotated[datetime, PlainSerializer(as_utc_iso, return_type=str)]

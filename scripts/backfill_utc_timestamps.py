"""
One-off backfill: convert job timestamps written in LOCAL time to UTC.

WHY
---
There used to be two clocks (see backend/app/timestamps.py for the full story).
Columns defaulted with `server_default=func.now()` are stamped by SQLite's
CURRENT_TIMESTAMP, which is UTC. Columns assigned in Python with
`datetime.now()` got the machine's LOCAL time. Both are naive, so nothing
recorded which was which, and one row could hold both:

    training_jobs id=8
      created_at  2026-07-20 07:13:20        <- UTC
      started_at  2026-07-20 12:43:20.571927 <- local (UTC+05:30)

The same instant, written five and a half hours apart. Read literally, that job
sat queued for most of a working day before starting.

New rows are written UTC throughout (`app.timestamps.utcnow`). This converts the
four columns that were previously assigned in Python:

    training_jobs.started_at    training_jobs.finished_at
    annotation_jobs.started_at  annotation_jobs.finished_at

Every other timestamp column already came from `func.now()` and is already UTC,
so this script deliberately does not touch them.

NOT URGENT. The app works either way — these columns feed a run's displayed
start/finish time, not any logic. Running it is what makes those displayed times
agree with everything else on the page.

HOW A ROW IS RECOGNISED AS UNCONVERTED
--------------------------------------
A naive timestamp carries no evidence of its own zone, so this uses the one
signal actually present in the data: within a single job row, `created_at` is
known-UTC, and a job is started by BackgroundTasks within seconds of being
queued. A `started_at` that appears MORE THAN AN HOUR after `created_at` is
therefore not a real delay — it is an unconverted local timestamp, offset by the
machine's distance from UTC.

That threshold is what makes the script safe to re-run: after conversion the gap
collapses to seconds and the row no longer matches. It also means a genuine
timezone west of UTC (a negative offset, where local time reads EARLIER than
UTC) is not detected by the gap test — those rows are found by the companion
check that `finished_at` precedes `started_at`, and are reported for review
rather than guessed at.

Conversion uses the system's offset AT EACH TIMESTAMP rather than one fixed
number, so a machine that observes DST converts a January row and a July row
correctly.

USAGE
-----
    ./backend/venv/Scripts/python.exe scripts/backfill_utc_timestamps.py --dry-run
    ./backend/venv/Scripts/python.exe scripts/backfill_utc_timestamps.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.database import SessionLocal, init_db  # noqa: E402
from app.models import AnnotationJob, TrainingJob  # noqa: E402
from sqlalchemy import select  # noqa: E402

#: A job is started by BackgroundTasks moments after it is queued. Anything
#: beyond this is a clock mismatch, not a queue delay.
IMPLAUSIBLE_QUEUE_DELAY = timedelta(hours=1)


def local_naive_to_utc_naive(value: datetime) -> datetime:
    """Reinterpret a naive LOCAL timestamp as naive UTC.

    `astimezone()` on a naive datetime assumes it is local time and attaches the
    offset in effect AT THAT DATE — which is why this handles DST correctly
    rather than subtracting one fixed number from every row.
    """
    return value.astimezone().astimezone(timezone.utc).replace(tzinfo=None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert local-time job timestamps to UTC.")
    parser.add_argument("--dry-run", action="store_true", help="report, don't write")
    args = parser.parse_args()

    init_db()

    offset = datetime.now().astimezone().utcoffset() or timedelta(0)
    print(f"system offset from UTC right now: {offset}")
    print(f"threshold for 'not a real queue delay': {IMPLAUSIBLE_QUEUE_DELAY}\n")

    db = SessionLocal()
    try:
        converted = 0
        suspicious: list[str] = []

        for model, label in ((TrainingJob, "training_job"), (AnnotationJob, "annotation_job")):
            for job in db.scalars(select(model)).all():
                if job.created_at is None:
                    continue

                # The gap test only looks at started_at; finished_at is written
                # by the same clock in the same row, so it moves with it. That
                # keeps a row internally consistent — converting one and not the
                # other would turn a wrong duration into an absurd one.
                started = job.started_at
                if started is None:
                    continue

                gap = started - job.created_at
                if gap <= IMPLAUSIBLE_QUEUE_DELAY:
                    # Already UTC, or a negative-offset machine — see below.
                    if job.finished_at is not None and job.finished_at < started:
                        suspicious.append(
                            f"{label} {job.id}: finished_at precedes started_at "
                            f"({job.finished_at} < {started})"
                        )
                    continue

                new_started = local_naive_to_utc_naive(started)
                new_finished = (
                    local_naive_to_utc_naive(job.finished_at)
                    if job.finished_at is not None
                    else None
                )
                print(
                    f"  {label} {job.id}: started {started} -> {new_started}"
                    + (f", finished {job.finished_at} -> {new_finished}" if new_finished else "")
                )
                if not args.dry_run:
                    job.started_at = new_started
                    if new_finished is not None:
                        job.finished_at = new_finished
                converted += 1

        if suspicious:
            print("\n  ! needs a human — not converted:")
            for line in suspicious:
                print(f"    {line}")

        if args.dry_run:
            db.rollback()
            print(f"\ndry run: {converted} row(s) would be converted")
        else:
            db.commit()
            print(f"\nconverted {converted} row(s)")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

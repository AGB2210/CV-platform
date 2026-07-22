"""
Live training logs — an in-memory ring buffer per job.

Training frameworks narrate usefully (dataset scan, epoch tables, warnings
about corrupt labels), but that narration goes to the server console where the
person who clicked Train never sees it. This module captures it per job so the
UI can show a live tail.

DELIBERATELY IN-MEMORY, deliberately bounded:

  - The DB row already stores the durable outcome (metrics per epoch, error
    with traceback). The log stream is *commentary* — worth watching live,
    not worth persisting. Writing every line to SQLite would add a write per
    log record to a database that also takes a commit per epoch.
  - One process serves everything (uvicorn, no workers), so a module-level
    dict IS the shared state. If jobs ever move to a separate worker process,
    this seam is where a Redis list would slot in.
  - A deque with maxlen keeps memory flat however chatty the framework gets;
    the UI only ever wants the tail anyway.

Capture works by attaching a logging.Handler to the framework's logger for the
duration of the run. Ultralytics routes its console narration through
logging ("ultralytics" logger), so a handler there sees the epoch tables and
warnings without touching sys.stdout — which is process-global and shared with
uvicorn, and must not be hijacked from a background thread.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from threading import Lock

#: Lines kept per job. Enough to scroll back through a few epochs of context;
#: the point is a live tail, not an archive.
_MAX_LINES = 400

_buffers: dict[int, deque[str]] = {}
_lock = Lock()


def append(job_id: int, line: str) -> None:
    """Add one line to a job's log. Creates the buffer on first use."""
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with _lock:
        buf = _buffers.setdefault(job_id, deque(maxlen=_MAX_LINES))
        # One line per call: embedded newlines would let a single record
        # masquerade as several, breaking the "lines" contract the UI renders.
        for part in str(line).splitlines():
            if part.strip():
                buf.append(f"[{stamp}] {part.rstrip()}")


def tail(job_id: int) -> list[str]:
    """The job's captured lines, oldest first. Empty if none were captured
    (job never ran in this process's lifetime, or was discarded)."""
    with _lock:
        buf = _buffers.get(job_id)
        return list(buf) if buf else []


def discard(job_id: int) -> None:
    """Drop a job's buffer — a cancelled run's narration has no audience."""
    with _lock:
        _buffers.pop(job_id, None)


class _JobLogHandler(logging.Handler):
    def __init__(self, job_id: int) -> None:
        super().__init__(level=logging.INFO)
        self.job_id = job_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            append(self.job_id, record.getMessage())
        except Exception:  # noqa: BLE001 — a log line must never hurt the run
            pass


class capture_framework_logs:
    """Context manager: route the training framework's log records into a
    job's buffer for the duration of a run.

    Attached to the framework loggers by NAME rather than the root logger, so
    uvicorn's request lines and our own app logging don't leak into a training
    log where they'd be noise.
    """

    #: Loggers worth listening to. Ultralytics uses "ultralytics"; torch's
    #: rare-but-important warnings ride on "py.warnings" when warnings are
    #: captured, which they aren't by default — keep the list short and real.
    LOGGER_NAMES = ("ultralytics",)

    def __init__(self, job_id: int) -> None:
        self.handler = _JobLogHandler(job_id)
        self._prior_levels: dict[str, int] = {}

    def __enter__(self) -> None:
        for name in self.LOGGER_NAMES:
            lg = logging.getLogger(name)
            lg.addHandler(self.handler)
            # The logger's EFFECTIVE level gates records before any handler
            # sees them; if the framework hasn't configured its logger yet
            # (or the root sits at WARNING), its INFO narration would be
            # filtered before capture. Open to INFO for the run, restore after.
            self._prior_levels[name] = lg.level
            if lg.getEffectiveLevel() > logging.INFO:
                lg.setLevel(logging.INFO)

    def __exit__(self, *exc: object) -> None:
        for name in self.LOGGER_NAMES:
            lg = logging.getLogger(name)
            lg.removeHandler(self.handler)
            lg.setLevel(self._prior_levels.get(name, logging.NOTSET))

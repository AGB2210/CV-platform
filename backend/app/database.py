"""
Database engine, session factory, and declarative base.

This module owns the SQLAlchemy plumbing. Route handlers never create sessions
themselves — they ask for one via the `get_db` dependency, which guarantees the
session is closed even if the handler raises.
"""

from collections.abc import Generator

import logging
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings
from app.timestamps import utcnow

# `check_same_thread=False` is a SQLite-specific requirement. By default SQLite
# refuses to use a connection from a thread other than the one that created it.
# FastAPI runs sync endpoints in a threadpool, so connections legitimately move
# between threads. SQLAlchemy's connection pool already serialises access, so
# relaxing this check is the documented, safe thing to do here.
logger = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # flip to True to see every SQL statement — useful while learning
)

@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Enable foreign key enforcement on every new SQLite connection.

    THIS IS NOT OPTIONAL, and it surprises almost everyone: SQLite ships with
    foreign key enforcement OFF by default, for backwards compatibility with
    versions that predate the feature. Without this, every `ondelete="CASCADE"`
    in our models is decorative — deleting a project leaves its images and
    categories behind as orphan rows pointing at an ID that no longer exists.

    The pragma is per-CONNECTION, not per-database, which is why this has to be
    an event listener rather than a one-off call at startup: the connection pool
    opens new connections over time, and each one starts with the pragma off.

    The `isinstance` guard exists because this listener is attached to the
    generic Engine class — if this project ever moves to Postgres (where FKs are
    always enforced), sending a SQLite pragma would error.
    """
    import sqlite3

    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")

        # WAL: readers and one writer can work at the SAME TIME.
        #
        # SQLite's default (journal_mode=delete) takes a lock over the whole
        # database for every write, so a reader arriving mid-write gets
        # "database is locked". This app writes from a background thread on a
        # cadence — the training runner commits after EVERY epoch, and the
        # annotation runner after every image — while the UI polls those same
        # tables every 1-2 seconds. That is the exact shape of contention WAL
        # exists for, and without it the failure surfaces as a random 500
        # partway through a long run.
        cursor.execute("PRAGMA journal_mode=WAL")

        # And when a write genuinely does have to wait, wait rather than fail
        # instantly. 15s comfortably covers the brief exclusive lock a commit
        # takes; the alternative is an error thrown while the answer was
        # milliseconds away.
        cursor.execute("PRAGMA busy_timeout=15000")

        # NORMAL is the standard companion to WAL: durable against application
        # crashes, and only at risk of losing the last transactions in an OS
        # crash or power cut. Worth it — FULL fsyncs on every commit, and this
        # commits once per training epoch.
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


# `sessionmaker` is a factory: calling SessionLocal() gives a fresh session.
# autoflush=False keeps SQLAlchemy from issuing surprise writes mid-transaction;
# we prefer explicit commits so the ordering of DB writes is easy to reason about.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class all ORM models inherit from.

    SQLAlchemy collects every subclass into `Base.metadata`, which is what lets
    `create_all()` know which tables to build. Phase 1 will add the first real
    models (Project, Image, Class) on top of this.
    """


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a request-scoped database session.

    One session per request is the standard pattern: work inside a request shares
    a transaction, and the `finally` block guarantees the connection returns to
    the pool even when a handler raises.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_missing_columns(bind) -> None:
    """Add columns that exist on the models but not in the database.

    THE PRE-ALEMBIC ERA'S MIGRATION STEP, kept only as the workhorse of the
    Alembic BASELINE revision (migrations/versions/0001_baseline.py), which
    runs it one final time to converge legacy databases. It is not called at
    startup any more — future schema changes are Alembic revisions.

    It only ever handled the one easy case (a new nullable-or-defaulted
    column) and needed six hand-written backfill scripts over its life for
    everything else — which is exactly why Alembic replaced it.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all already handled it

        live_columns = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in live_columns:
                continue

            # SQLite requires a non-NULL default when adding a NOT NULL
            # column to a table with existing rows — there's no other value
            # it could give them.
            type_sql = column.type.compile(dialect=bind.dialect)
            clause = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {type_sql}"

            default = column.default.arg if column.default is not None else None
            if not column.nullable:
                if default is None:
                    # Can't invent a value. Skip rather than corrupt the
                    # table, and say so loudly.
                    print(
                        f"  ! cannot auto-add NOT NULL column "
                        f"{table.name}.{column.name} without a default — "
                        f"delete the DB or write a migration"
                    )
                    continue
                literal = f"'{default}'" if isinstance(default, str) else int(default)
                clause += f" NOT NULL DEFAULT {literal}"
            elif default is not None:
                literal = f"'{default}'" if isinstance(default, str) else default
                clause += f" DEFAULT {literal}"

            print(f"  + {table.name}.{column.name}")
            bind.execute(text(clause))


def _run_migrations() -> None:
    """Bring the database to the newest schema revision via Alembic.

    Programmatic `upgrade head` rather than shelling out: the same venv, the
    same engine, no PATH questions. The baseline revision (0001) converges
    any pre-Alembic database — fresh, legacy, or already-stamped, every
    startup lands on the same known state, and future schema changes are
    ordinary revisions in backend/migrations/versions/.
    """
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    backend_dir = Path(__file__).resolve().parent.parent
    cfg = Config(str(backend_dir / "alembic.ini"))
    # Absolute, so it works whatever the process's CWD is (the launcher runs
    # uvicorn from backend/, tests may run from anywhere).
    cfg.set_main_option("script_location", str(backend_dir / "migrations"))
    command.upgrade(cfg, "head")


def init_db() -> None:
    """Migrate the schema to head, then clear orphaned jobs."""
    # Importing the models package registers every model class on Base.metadata
    # — the baseline migration builds tables from exactly that.
    from app import models  # noqa: F401  (imported for side effect)

    _run_migrations()
    _fail_interrupted_jobs()


def _fail_interrupted_jobs() -> None:
    """Close out jobs that were running when the process died.

    Jobs run in a background thread of THIS process. If it's killed — Ctrl+C,
    a crash, a reboot, or the launcher's own reaper — the thread dies with it,
    but the row is left saying `running` forever. Nothing else ever reconciles
    that: the runner is gone, so the code that would set a terminal status is
    gone too.

    The visible symptom is a UI that polls a job which will never move again —
    and since GPU admission (services/gpu_admission.py) reads RUNNING rows
    globally to decide whether the card is busy, one orphaned row would make
    every future job wait for a GPU that nothing is actually using.

    Startup is the one moment we can be certain: no job can legitimately be
    running yet, because nothing has had a chance to start one. So anything
    found in a live state is by definition a leftover.
    """
    from sqlalchemy import select

    from app.models import AnnotationJob, EvaluationJob, JobStatus, TrainingJob

    with SessionLocal() as db:
        reclaimed = 0
        for model in (TrainingJob, AnnotationJob, EvaluationJob):
            for job in db.scalars(
                select(model).where(
                    model.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
                )
            ).all():
                # A job the user had already asked to cancel ends as CANCELLED,
                # not failed — the restart merely delivered what was requested.
                # Without this, cancel-then-close-the-window reported "failed"
                # for a run the user deliberately stopped. Applies to any job
                # type with a control column (training and annotation).
                if getattr(job, "control", None) == "cancel":
                    job.status = JobStatus.CANCELLED
                    job.control = None
                    if hasattr(job, "status_detail"):
                        job.status_detail = None
                    job.finished_at = utcnow()
                    reclaimed += 1
                    continue
                job.status = JobStatus.FAILED
                job.error = (
                    "The server stopped while this job was running, so it was "
                    "interrupted. Nothing was corrupted — start it again."
                )
                # Clear any admission wait note (training/annotation jobs) —
                # it referred to a world that no longer exists.
                if hasattr(job, "status_detail"):
                    job.status_detail = None
                job.finished_at = utcnow()
                reclaimed += 1
        if reclaimed:
            db.commit()
            logger.warning(
                "Marked %d interrupted job(s) as failed at startup", reclaimed
            )

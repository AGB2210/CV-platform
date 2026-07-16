"""
Database engine, session factory, and declarative base.

This module owns the SQLAlchemy plumbing. Route handlers never create sessions
themselves — they ask for one via the `get_db` dependency, which guarantees the
session is closed even if the handler raises.
"""

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# `check_same_thread=False` is a SQLite-specific requirement. By default SQLite
# refuses to use a connection from a thread other than the one that created it.
# FastAPI runs sync endpoints in a threadpool, so connections legitimately move
# between threads. SQLAlchemy's connection pool already serialises access, so
# relaxing this check is the documented, safe thing to do here.
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


def init_db() -> None:
    """Create any tables that don't exist yet.

    `create_all` only issues CREATE TABLE for missing tables — it will NOT alter
    an existing table to match a changed model. That's fine while the schema is
    still in flux (delete the .db file and restart to reset). Once there's data
    worth keeping, this is the seam where Alembic migrations would slot in.
    """
    # Importing the models package registers every model class on Base.metadata.
    # Without this import, create_all() would find nothing to create.
    from app import models  # noqa: F401  (imported for side effect)

    Base.metadata.create_all(bind=engine)

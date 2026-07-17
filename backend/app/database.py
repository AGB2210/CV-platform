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


def _add_missing_columns() -> None:
    """Add columns that exist on the models but not yet in the database.

    WHY THIS EXISTS
    ---------------
    `create_all()` only issues CREATE TABLE for MISSING tables. It will not
    touch a table that already exists, so adding a field to a model is silently
    a no-op against an existing database — the app then crashes on the first
    query with "no such column", which is a confusing way to learn this.

    This is a deliberately tiny migration step covering the one case that
    actually comes up while a schema is still in flux: a new nullable-or-
    defaulted column. It reads the live schema, diffs it against the model, and
    issues ALTER TABLE ADD COLUMN for anything absent.

    WHAT IT DELIBERATELY DOES NOT DO
    --------------------------------
    Dropped columns, renames, type changes, new constraints, or backfills that
    need real logic. SQLite can't do most of those without rebuilding the table
    anyway. The moment one of those is needed, this should be deleted and
    replaced with Alembic — which is the real answer, and is exactly what this
    function is standing in for. It is a convenience, not a migration system.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all will handle it

            live_columns = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in live_columns:
                    continue

                # SQLite requires a non-NULL default when adding a NOT NULL
                # column to a table with existing rows — there's no other value
                # it could give them.
                type_sql = column.type.compile(dialect=engine.dialect)
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
                conn.execute(text(clause))


def init_db() -> None:
    """Create missing tables, then add any missing columns.

    Once there's data worth keeping, `_add_missing_columns` should be replaced
    by Alembic — see its docstring.
    """
    # Importing the models package registers every model class on Base.metadata.
    # Without this import, create_all() would find nothing to create.
    from app import models  # noqa: F401  (imported for side effect)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()

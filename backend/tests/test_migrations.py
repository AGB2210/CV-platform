"""
Alembic wiring — the baseline revision converges every database state.

Three databases exist in the wild: fresh (nothing), legacy (created by the
pre-Alembic create_all + add-missing-columns era, possibly missing recent
columns), and current (already stamped). `upgrade head` must land all three on
the same schema, stamped, and be a no-op when run twice.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text

BACKEND = Path(__file__).resolve().parent.parent


def _upgrade(engine) -> None:
    """Run `upgrade head` against a specific (throwaway) engine."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    with engine.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        conn.commit()


def _tmp_engine(tmp_path, name: str):
    return create_engine(
        f"sqlite:///{(tmp_path / name).as_posix()}",
        connect_args={"check_same_thread": False},
    )


def test_fresh_database_migrates_to_full_schema(tmp_path):
    from app.database import Base
    import app.models  # noqa: F401

    engine = _tmp_engine(tmp_path, "fresh.db")
    _upgrade(engine)

    tables = set(inspect(engine).get_table_names())
    expected = {t.name for t in Base.metadata.sorted_tables}
    assert expected <= tables, f"missing: {expected - tables}"
    assert "alembic_version" in tables, "the DB must be stamped"

    # Idempotent: a second upgrade is a clean no-op.
    _upgrade(engine)


def test_legacy_database_gains_missing_columns(tmp_path):
    """A database from the pre-Alembic era: tables exist but recent columns
    don't. The baseline's add-missing-columns pass fills them in."""
    import app.models  # noqa: F401

    engine = _tmp_engine(tmp_path, "legacy.db")
    # A minimal old-era training_jobs table: several modern columns absent
    # (status_detail, stopped_early arrived late), plus a row that must survive.
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE training_jobs ("
                " id INTEGER PRIMARY KEY,"
                " project_id INTEGER NOT NULL,"
                " trainer_key VARCHAR(64) NOT NULL,"
                " status VARCHAR(16) NOT NULL,"
                " epochs INTEGER NOT NULL DEFAULT 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO training_jobs (project_id, trainer_key, status, epochs)"
                " VALUES (1, 'yolo', 'done', 5)"
            )
        )

    _upgrade(engine)

    cols = {c["name"] for c in inspect(engine).get_columns("training_jobs")}
    assert "status_detail" in cols, "late-era column must be added"
    assert "stopped_early" in cols

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT trainer_key, epochs FROM training_jobs")
        ).one()
        assert tuple(row) == ("yolo", 5), "existing data must survive the migration"

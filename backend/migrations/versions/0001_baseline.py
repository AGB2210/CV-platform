"""Baseline: converge any pre-1.0 database onto the current schema.

Revision ID: 0001
Revises: None

THE TRANSITION MIGRATION. Before Alembic, the schema was maintained by
create_all() (new tables) plus a homegrown add-missing-columns step — which
worked, but needed six hand-written backfill scripts over its life and could
never do renames, drops or constraint changes. This revision exists so every
database that ever ran a pre-Alembic build lands on the same, known state:

  - a FRESH database gets every table created from the live metadata;
  - a LEGACY database gets its missing tables created and its missing
    columns added — exactly what the old startup step did, run one last time;
  - either way it is then stamped `0001`, and all FUTURE schema changes are
    ordinary Alembic revisions on top.

Deliberately built from the LIVE metadata rather than a frozen snapshot of
DDL: pre-1.0 nothing has shipped, so "the current models" and "the baseline
schema" are the same thing by definition. The first post-1.0 revision breaks
that equivalence, which is precisely why Alembic is being adopted now.

No downgrade: there is nothing below the baseline to go back to.
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from app.database import Base, add_missing_columns
    from app import models  # noqa: F401 — registers every model on Base.metadata

    bind = op.get_bind()
    # create_all skips tables that already exist — fresh DBs get everything,
    # legacy DBs only what they lack.
    Base.metadata.create_all(bind=bind)
    # …and legacy tables get their missing columns, the old startup step's job.
    add_missing_columns(bind)


def downgrade() -> None:
    raise NotImplementedError("There is nothing below the baseline.")

"""Imported weights: the table, and the training-job column referencing one.

Revision ID: 0002
Revises: 0001

The first ordinary revision after the baseline. Adds the third "Initialize
from" source for training runs: a checkpoint uploaded from outside the app
(models/imported_weights.py), plus `training_jobs.init_weights_id` pointing
at it.

GUARDED, because databases arrive here from two different pasts: one stamped
`0001` long ago (needs both changes applied), and one created fresh today
(revision 0001 builds from LIVE metadata, so it already created both — doing
so again would fail). Checking what actually exists converges both.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "imported_weights" not in inspector.get_table_names():
        op.create_table(
            "imported_weights",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "project_id",
                sa.Integer(),
                sa.ForeignKey("projects.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("filename", sa.String(255), nullable=False),
            sa.Column("stored_path", sa.String(500), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("sha256", sa.String(64), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
            ),
        )
        op.create_index(
            "ix_imported_weights_project_id", "imported_weights", ["project_id"]
        )
        op.create_index("ix_imported_weights_sha256", "imported_weights", ["sha256"])

    columns = {c["name"] for c in inspector.get_columns("training_jobs")}
    if "init_weights_id" not in columns:
        op.add_column(
            "training_jobs", sa.Column("init_weights_id", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    op.drop_column("training_jobs", "init_weights_id")
    op.drop_table("imported_weights")

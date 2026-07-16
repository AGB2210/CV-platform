"""
ORM models package.

Empty in Phase 0 — the schema arrives in Phase 1 (Project, Image, Class).

`database.init_db()` imports this package so that importing a model module here
registers it on `Base.metadata`. When Phase 1 adds `project.py`, it gets wired in
with a single line below, e.g.:

    from app.models.project import Project  # noqa: F401
"""

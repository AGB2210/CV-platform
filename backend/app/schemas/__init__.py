"""
Pydantic schemas package — the API's request/response contracts.

Kept deliberately separate from `app.models` (SQLAlchemy). They describe
different things and change for different reasons:

  - models  = how data is stored (columns, indexes, foreign keys)
  - schemas = what the API accepts and returns (validation, field exposure)

Collapsing the two is a common shortcut that hurts later: every DB column
becomes public API by accident, and you can't change storage without changing
your contract. Phase 1 adds the first schemas here.
"""

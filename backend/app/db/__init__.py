"""DB package: SQLModel engine/session factory + lightweight migrations.

Re-exports the public surface from `session.py` so existing call sites keep
using `from app.db import get_session, session_scope, init_db`.
"""

from app.db.session import get_engine, get_session, init_db, session_scope

__all__ = ["get_engine", "get_session", "init_db", "session_scope"]

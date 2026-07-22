from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings.ensure_dirs()
        _engine = create_engine(
            f"sqlite:///{settings.db_path}",
            connect_args={"check_same_thread": False},
        )
        with _engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    return _engine


def init_db() -> None:
    # Import models so SQLModel.metadata knows every table
    from app import models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())


def get_session():
    with Session(get_engine()) as session:
        yield session


@contextmanager
def session_scope():
    with Session(get_engine()) as session:
        yield session

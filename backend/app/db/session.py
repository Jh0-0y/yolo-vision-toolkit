from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings

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


def _add_missing_columns() -> None:
    """Lightweight migration: create_all can't ALTER existing tables, so add
    columns introduced after a table was first created. New columns are nullable
    so existing rows keep NULL (e.g. project_id NULL = legacy/shared)."""
    additions = {
        "model_entry": [("project_id", "VARCHAR")],
        "train_run": [("project_id", "VARCHAR")],
        "lab_project": [("preset_json", "VARCHAR")],
    }
    engine = get_engine()
    with engine.begin() as conn:
        for table, cols in additions.items():
            existing = {
                row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")
            }
            if not existing:
                continue  # table not created yet — create_all will include the column
            for name, coltype in cols:
                if name not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")


def _relocate_scoped_files() -> None:
    """Move files of scoped models/runs from the flat pool into their project
    folder. Idempotent: only moves when the flat dir still exists and the target
    doesn't (so already-relocated or shared/NULL items are left alone)."""
    import shutil

    from sqlmodel import select

    from app.core.config import settings
    from app.models import ModelEntry, TrainRun

    def _move(flat, target):
        if flat.exists() and flat != target and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat), str(target))

    with Session(get_engine()) as session:
        for m in session.exec(select(ModelEntry)).all():
            if m.project_id:
                _move(settings.models_dir / m.id, settings.model_dir(m.project_id, m.id))
        for r in session.exec(select(TrainRun)).all():
            if r.project_id:
                _move(settings.runs_dir / r.id, settings.run_dir(r.project_id, r.id))


def init_db() -> None:
    # Import models so SQLModel.metadata knows every table
    from app import models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())
    _add_missing_columns()
    _relocate_scoped_files()


def get_session():
    with Session(get_engine()) as session:
        yield session


@contextmanager
def session_scope():
    with Session(get_engine()) as session:
        yield session

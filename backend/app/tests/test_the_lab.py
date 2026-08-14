"""연구실은 하나뿐이다 — 그 하나를 집어내는 규칙.

목록도 생성도 없으므로 `the_lab()` 이 유일한 진입점이다. 여기서 지키는 것은 둘이다:
**id 는 절대 안 바뀐다**(디스크의 영상과 크롭 런이 그 아래 있다), **이름은 코드가 정한다**
(바꿀 UI 가 없으니 DB 에 남은 옛 이름을 붙들 이유가 없다).
"""

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.api.v1.endpoints.labs import LAB_NAME, the_lab
from app.core.config import settings
from app.models import LabProject


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_first_call_creates_the_one_lab(session, tmp_path):
    lab = the_lab(session)

    assert lab.name == LAB_NAME
    assert session.exec(select(LabProject)).all() == [lab]
    assert (tmp_path / "labs" / lab.id).is_dir()


def test_calling_again_reuses_it_instead_of_making_another(session):
    first = the_lab(session)
    second = the_lab(session)

    assert first.id == second.id
    assert len(session.exec(select(LabProject)).all()) == 1


def test_an_existing_lab_keeps_its_id(session):
    """id 를 갈아치우면 지난 영상과 크롭 런이 통째로 안 보이게 된다."""
    existing = LabProject(name="whatever")
    session.add(existing)
    session.commit()

    assert the_lab(session).id == existing.id


def test_the_name_is_taken_from_the_code(session):
    """이름을 바꿀 화면이 없다 — DB 에 남은 옛 이름은 코드 값으로 맞춘다."""
    session.add(LabProject(name="test"))
    session.commit()

    assert the_lab(session).name == LAB_NAME


def test_the_marker_file_follows_the_name(session, tmp_path):
    import json

    lab = the_lab(session)
    marker = json.loads((tmp_path / "labs" / lab.id / "lab.json").read_text())

    assert marker["name"] == LAB_NAME


def test_the_oldest_lab_wins_if_several_somehow_exist(session):
    """야구 연구실이 생기면 코드로 들어온다 — 그전까지 여러 행이 생길 일은 없지만,
    생기더라도 매번 다른 연구실을 집으면 안 된다."""
    from datetime import datetime, timezone

    old = LabProject(name="a", created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    new = LabProject(name="b", created_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    session.add(old)
    session.add(new)
    session.commit()

    assert the_lab(session).id == old.id

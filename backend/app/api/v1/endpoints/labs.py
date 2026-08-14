"""연구실 — **하나뿐이다.**

연구실은 학습실 프로젝트와 다르다. 프로젝트는 도메인마다 늘어나지만(농구·야구),
연구실은 *연구 하나*다. 지금은 `basketball-adaptive-crop` 이고, 나중에 야구를 하게
되면 `baseball-adaptive-crop` 이 생기는데 그건 패키지부터 다른 별개의 파이프라인이라
**런타임에 만드는 것이 아니라 코드로 들어온다.**

그래서 목록도 생성도 삭제도 없다. 여기서 하는 일은 "그 하나를 돌려주는 것"뿐이다.
안에 영상 아카이브와 크롭 런이 있고, 라벨·클래스·데이터셋은 갖지 않는다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session, select

from app.db import get_session
from app.models import LabProject, iso_utc
from app.schemas.lab import LabOut
from app.services import lab_store

router = APIRouter(prefix="/lab", tags=["lab"])

# 이 연구실이 무엇인지 — 쓰는 파이프라인(adaptive-crop)과 종목을 그대로 이름에 둔다.
# 야구를 하게 되면 `baseball-adaptive-crop` 이 **코드로** 들어온다(패키지부터 다르다).
LAB_NAME = "basketball-adaptive-crop"
LAB_DESCRIPTION = (
    "Follows the ball and players through a wide broadcast feed and cuts a vertical crop."
)


def the_lab(session: Session) -> LabProject:
    """하나뿐인 연구실. 없으면 만든다 — 첫 실행에도 화면이 비지 않게.

    **id 는 절대 갈아치우지 않는다** — 디스크의 영상과 크롭 런이 그 id 아래 있어서,
    바꾸면 지난 연구가 통째로 안 보이게 된다. 반대로 **이름은 코드가 정한다**:
    바꿀 UI 가 없으니 DB 에 남은 옛 이름을 붙들고 있을 이유가 없다.
    """
    lab = session.exec(select(LabProject).order_by(LabProject.created_at)).first()
    fresh = lab is None
    if lab is None:
        lab = LabProject(name=LAB_NAME)
        lab_store.ensure_dirs(lab.id)
    if fresh or lab.name != LAB_NAME:
        lab.name = LAB_NAME
        lab_store.write_marker(lab.id, LAB_NAME)
        session.add(lab)
        session.commit()
        session.refresh(lab)
    return lab


def _out(lab: LabProject) -> LabOut:
    return LabOut(
        id=lab.id,
        name=lab.name,
        description=LAB_DESCRIPTION,
        created_at=iso_utc(lab.created_at),
        video_count=len(lab_store.list_videos(lab.id)),
        run_count=lab_store.count_runs(lab.id),
    )


@router.get("", response_model=LabOut)
async def get_the_lab(session: Session = Depends(get_session)):
    lab = the_lab(session)
    return await run_in_threadpool(_out, lab)

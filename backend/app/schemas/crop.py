"""크롭 랩 산출물(crop run) DTO."""

from pydantic import BaseModel


class CropRunOut(BaseModel):
    id: str
    name: str = ""
    kind: str = "json"  # "json" = 좌표만 | "cut" = 추론 없는 크롭 컷
    source_name: str = ""
    created_at: str = ""
    # progress.jsonl 에서 파생 — 저장되는 값이 아니다
    status: str = "running"  # running | done | error | cancelled
    error: str | None = None
    settings: dict = {}  # 그때 쓴 튜닝 노브·검출기 스냅샷
    summary: dict | None = None  # crop.json 의 커버리지 요약
    has_crop_json: bool = False
    has_video: bool = False
    video_expired: bool = False


class CropRunRename(BaseModel):
    name: str

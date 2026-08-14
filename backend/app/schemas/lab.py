"""연구실(Lab) 요청·응답 DTO."""

from pydantic import BaseModel


class LabOut(BaseModel):
    id: str
    name: str
    description: str = ""  # 무슨 연구인지 — 코드가 정한다
    created_at: str
    video_count: int = 0
    run_count: int = 0


class LabVideoOut(BaseModel):
    id: str
    name: str  # 사용자에게 보이는 이름 (기본값 = 업로드한 파일명)
    filename: str  # 원래 업로드 파일명
    ext: str
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_ms: int = 0
    created_at: str = ""
    run_count: int = 0  # 이 영상으로 돌린 크롭 런 수


class LabVideoPatch(BaseModel):
    name: str

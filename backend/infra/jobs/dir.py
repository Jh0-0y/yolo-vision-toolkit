"""잡 디렉터리 — 진행률 파일과 취소 센티널이 함께 사는 곳.

둘 다 `<root>/<job_id>/` 안에 있으므로, 경로를 매번 조립하는 대신 이 값 객체를
들고 다닌다. API 프로세스와 워커 프로세스가 같은 객체를 쓴다 — 워커는 설정을
모르므로 루트 경로를 인자로 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from infra.jobs import cancel, progress


@dataclass(frozen=True)
class JobDir:
    path: Path

    @property
    def progress_path(self) -> Path:
        return progress.path_in(self.path)

    @property
    def cancel_path(self) -> Path:
        return cancel.path_in(self.path)

    def ensure(self) -> JobDir:
        """디렉터리만 만든다. 워커가 자기 잡 디렉터리를 확보할 때."""
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    def prepare(self) -> JobDir:
        """새 잡을 받을 준비 — 디렉터리 생성 · 빈 진행률 파일 · 묵은 취소 신호 제거.
        잡을 제출하는 쪽(API 프로세스)이 부른다."""
        self.ensure()
        self.progress_path.touch()
        cancel.clear(self.cancel_path)
        return self

    def reset(self) -> JobDir:
        """진행률을 비우고 다시 시작한다. 같은 대상을 재실행할 때
        (예: 프레임 재추출) 이전 이벤트가 섞이지 않게."""
        self.ensure()
        self.progress_path.write_text("")
        cancel.clear(self.cancel_path)
        return self

    def emit(self, event: dict) -> None:
        progress.emit(self.progress_path, event)

    def read(self, offset: int = 0) -> tuple[list[dict], int]:
        return progress.read(self.progress_path, offset)

    def request_cancel(self) -> None:
        cancel.request(self.cancel_path)

    def cancelled(self) -> bool:
        return cancel.is_set(self.cancel_path)


def at(root: Path, job_id: str) -> JobDir:
    """`root/<job_id>` 잡 디렉터리."""
    return JobDir(root / job_id)

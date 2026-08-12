"""잡 취소 신호 — 파일 하나의 존재 여부가 곧 신호다.

프로세스 경계를 넘는 유일한 취소 수단이다. API 가 파일을 만들면 워커가 루프마다
존재를 확인해 스스로 멈춘다. 자식을 강제로 죽이지 않으므로 워커가 정리(임시 파일
삭제 등)를 마칠 기회를 갖는다.
"""

from __future__ import annotations

from pathlib import Path

FILENAME = "CANCEL"


def path_in(job_dir: Path) -> Path:
    return job_dir / FILENAME


def request(cancel_path: Path) -> None:
    """취소를 요청한다. 워커가 다음 확인 지점에서 멈춘다."""
    cancel_path.touch()


def clear(cancel_path: Path) -> None:
    """묵은 신호를 지운다. 새 잡을 시작하기 전에 반드시 부른다 —
    잡 디렉터리를 재사용하면 이전 취소가 새 잡을 즉시 죽인다."""
    cancel_path.unlink(missing_ok=True)


def is_set(cancel_path: Path) -> bool:
    return cancel_path.exists()

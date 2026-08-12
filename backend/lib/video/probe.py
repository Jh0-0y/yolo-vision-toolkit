"""영상의 기본 정보 읽기 — 크기 · fps · 프레임 수."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int  # 컨테이너가 값을 안 들고 있으면 0

    @property
    def size(self) -> tuple[int, int]:
        """cv2.VideoWriter 가 받는 (width, height) 순서."""
        return self.width, self.height

    @property
    def duration_ms(self) -> int:
        """프레임 수 기준 길이. 프레임 수를 모르면 0 —
        그때는 호출자가 다른 근거(예: 마지막 샘플 시각)를 써야 한다."""
        if self.frame_count <= 0 or not self.fps:
            return 0
        return int(self.frame_count / self.fps * 1000)


def probe(path: Path | str) -> VideoMeta:
    """열었다 바로 닫으며 규격만 읽는다. 열지 못하면 ValueError.

    못 여는 파일에 cv2 는 **0 이 아니라 -1** 을 돌려준다. -1 은 truthy 라서
    `or 25.0` 류의 폴백을 그대로 통과해 음수 해상도가 아래로 흘러간다 — 그래서
    값을 읽기 전에 isOpened() 로 막는다.

    fps 는 컨테이너가 값을 안 들고 있을 때(0) 25.0 으로 본다(cv2 관례). 0 을
    그대로 쓰면 시각 계산이 전부 0으로 나눠진다.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        return VideoMeta(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
            frame_count=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
    finally:
        cap.release()

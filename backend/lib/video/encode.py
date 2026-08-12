"""브라우저에서 재생 가능한 H.264 mp4 로 트랜스코딩.

cv2.VideoWriter 의 H.264 지원은 OpenCV 빌드마다 달라 믿을 수 없다. 그래서 항상
mp4v 로 먼저 쓰고 여기서 ffmpeg 로 다시 인코딩한다. ffmpeg 는 선택이 아니라
필수다 — 없으면 재생 불가능한 파일을 남기는 대신 잡을 실패시킨다.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_MISSING = (
    "ffmpeg is required to encode a browser-playable video but was not found. "
    "Install it (Docker image ships it; locally: `brew install ffmpeg`)."
)


def require_ffmpeg() -> None:
    """없으면 즉시 실패한다. 무거운 계산을 시작하기 **전에** 부른다 —
    한 시간 인코딩한 뒤 마지막 단계에서 실패하는 걸 막으려는 것."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(_MISSING)


def to_h264(src: Path, dst: Path) -> None:
    """H.264/yuv420p mp4 로 변환한다. 호출 전에 require_ffmpeg() 로 확인한다."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            # yuv420p 는 짝수 해상도를 요구한다 — 크롭으로 홀수가 됐으면 1px 채운다
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

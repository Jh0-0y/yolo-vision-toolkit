"""lib/video 계약 테스트 — 규격 읽기와 ffmpeg 선행 확인.

세 워커가 각자 열던 VideoCapture 를 한 함수로 모은 자리라, fps 폴백과
frame_count=0 처리를 여기서 고정해 둔다.
"""

from __future__ import annotations

import pytest

from lib import video
from lib.video import encode


def _write_sample(path, frames: int, size=(64, 48), fps: float = 10.0) -> bool:
    """mp4v 로 짧은 영상을 만든다. 이 OpenCV 빌드가 못 쓰면 False."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        return False
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i * 5 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.exists() and path.stat().st_size > 0


def test_duration_ms_from_frame_count():
    meta = video.VideoMeta(width=1920, height=1080, fps=25.0, frame_count=50)

    assert meta.duration_ms == 2000
    assert meta.size == (1920, 1080)


def test_duration_ms_is_zero_when_frame_count_unknown():
    """컨테이너가 프레임 수를 안 들고 있으면 0 — 호출자가 다른 근거를 쓴다."""
    meta = video.VideoMeta(width=1920, height=1080, fps=25.0, frame_count=0)

    assert meta.duration_ms == 0


def test_probe_reads_geometry(tmp_path):
    path = tmp_path / "sample.mp4"
    if not _write_sample(path, frames=12):
        pytest.skip("this OpenCV build cannot write mp4v")

    meta = video.probe(path)

    assert (meta.width, meta.height) == (64, 48)
    assert meta.fps == pytest.approx(10.0, abs=0.5)
    assert meta.frame_count == 12


def test_probe_rejects_an_unreadable_file(tmp_path):
    """cv2 는 못 여는 파일에 -1 을 돌려준다. 음수 해상도가 아래로 흘러가면
    VideoWriter 가 조용히 깨진 파일을 만드므로 여기서 끊는다."""
    broken = tmp_path / "not-a-video.mp4"
    broken.write_bytes(b"nope")

    with pytest.raises(ValueError, match="Cannot open video"):
        video.probe(broken)


def test_require_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr(encode.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        video.require_ffmpeg()


def test_require_ffmpeg_passes_when_present(monkeypatch):
    monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    video.require_ffmpeg()  # 예외 없음

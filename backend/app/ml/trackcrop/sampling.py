"""Frame Sampling — SAMPLING_INTERVAL_MS(100ms) 간격 프레임 추출."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .constants import SAMPLING_INTERVAL_MS
from .errors import ErrorCode, TrackCropError


def sample_frames(
    path: Path, interval_ms: int = SAMPLING_INTERVAL_MS
) -> Iterator[tuple[int, np.ndarray]]:
    """(video_offset_ms, frame BGR ndarray)를 interval_ms 격자로 생성한다.

    순차 읽기 방식 — 목표 offset(0, interval, 2·interval, …)에 도달한 첫 프레임을 그
    offset의 대표 프레임으로 낸다. seek 반복보다 빠르고 프레임 누락이 없다.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise TrackCropError(
            ErrorCode.VIDEO_DECODE_FAILED,
            "영상을 열 수 없습니다.",
            details={"path": str(path)},
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise TrackCropError(
            ErrorCode.VIDEO_DECODE_FAILED,
            "FPS를 확인할 수 없습니다.",
            details={"path": str(path)},
        )

    try:
        next_offset_ms = 0
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_ms = frame_index / fps * 1000
            if frame_ms + (1000 / fps) / 2 >= next_offset_ms:
                yield next_offset_ms, frame
                next_offset_ms += interval_ms
            frame_index += 1
    finally:
        cap.release()

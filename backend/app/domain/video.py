"""Video frame extraction for auto-labeling ingestion.

Samples frames from a video into a project's ``raw/`` directory so the rest of
the labeling pipeline (thumbnails, inference, review) can reuse them unchanged.
CPU/IO only — never touches the GPU, so it runs off the GPU-serialized executor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.domain.tiling import TilingParams, tile_grid, tile_stem

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class ExtractParams:
    target_fps: float = 2.0
    max_frames: int = 2000
    start_sec: float = 0.0
    end_sec: float | None = None
    dedup: bool = True
    dedup_threshold: float = 0.92  # >= means "too similar", frame is skipped
    # 타일링 — 프레임을 학습용 타일로 쪼개 저장 (라벨 전 단계라 가시 비율 없음)
    tile: bool = False
    tile_size: int = 640
    stride: int = 480


def _similarity(a, b) -> float:
    """1.0 == identical, 0.0 == maximally different (mean abs diff on 32x32 gray)."""
    diff = cv2.absdiff(a, b).mean() / 255.0
    return 1.0 - float(diff)


def extract_frames(
    video_path: Path,
    raw_dir: Path,
    stem: str,
    params: ExtractParams,
    emit: Callable[[dict], None],
    cancel_path: Path,
) -> dict:
    """Extract sampled frames into ``raw_dir``. Returns a summary dict.

    진행 상황은 ``emit`` 으로만 알린다 — 어디에 기록할지는 호출자가 정한다.
    잡 시스템을 몰라야 웹 없이도(CLI·배치) 이 함수를 쓸 수 있다.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        emit({"phase": "error", "msg": "Cannot open video"})
        raise ValueError("Cannot open video")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # sampling step: how many source frames per saved frame
    step = max(1, round(src_fps / params.target_fps)) if src_fps > 0 else 1
    start_frame = int(params.start_sec * src_fps) if src_fps > 0 else 0
    end_frame = (
        int(params.end_sec * src_fps)
        if (params.end_sec is not None and src_fps > 0)
        else total_frames or None
    )

    emit({
        "phase": "start",
        "src_fps": round(src_fps, 2),
        "total_frames": total_frames,
        "step": step,
    })

    idx = 0
    saved = 0
    tiles = 0
    skipped_dup = 0
    prev_small = None
    cancelled = False

    try:
        while True:
            if cancel_path.exists():
                cancelled = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            if idx < start_frame:
                idx += 1
                continue
            if end_frame is not None and idx >= end_frame:
                break

            if (idx - start_frame) % step == 0:
                keep = True
                if params.dedup:
                    small = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)
                    )
                    if prev_small is not None and _similarity(small, prev_small) >= params.dedup_threshold:
                        keep = False
                        skipped_dup += 1
                    else:
                        prev_small = small

                if keep:
                    # Sequential, contiguous frame numbering: {video-name}_00001.jpg,
                    # _00002.jpg ... in save order (not the sparse source frame index).
                    frame_stem = f"{stem}_{saved + 1:05d}"
                    if params.tile:
                        h, w = frame.shape[:2]
                        grid = TilingParams(tile_size=params.tile_size, stride=params.stride)
                        for col, row, tx, ty in tile_grid(w, h, grid):
                            crop = frame[ty : ty + grid.tile_size, tx : tx + grid.tile_size]
                            cv2.imwrite(str(raw_dir / f"{tile_stem(frame_stem, col, row)}.jpg"), crop)
                            tiles += 1
                    else:
                        cv2.imwrite(str(raw_dir / f"{frame_stem}.jpg"), frame)
                    saved += 1
                    if saved % 10 == 0 or saved == 1:
                        emit({
                            "phase": "extract",
                            "saved": saved,
                            "scanned": idx + 1,
                            "total_frames": total_frames,
                            "skipped_dup": skipped_dup,
                        })
                    if saved >= params.max_frames:
                        break
            idx += 1
    finally:
        cap.release()

    if cancelled:
        emit({"phase": "cancelled", "saved": saved, "tiles": tiles})
        return {"status": "cancelled", "saved": saved, "tiles": tiles, "skipped_dup": skipped_dup}

    emit({"phase": "done", "saved": saved, "tiles": tiles, "scanned": idx, "skipped_dup": skipped_dup})
    return {"status": "done", "saved": saved, "tiles": tiles, "skipped_dup": skipped_dup, "scanned": idx}

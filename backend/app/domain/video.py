"""Video frame extraction for auto-labeling ingestion.

Samples frames from a video into a project's ``raw/`` directory so the rest of
the labeling pipeline (thumbnails, inference, review) can reuse them unchanged.
CPU/IO only — never touches the GPU, so it runs off the GPU-serialized executor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class ExtractParams:
    target_fps: float = 2.0
    max_frames: int = 2000
    start_sec: float = 0.0
    end_sec: float | None = None
    dedup: bool = True
    dedup_threshold: float = 0.92  # >= means "too similar", frame is skipped


def extract_scrub_frames(video_path: Path, out_dir: Path, max_frames: int = 150) -> int:
    """Uniformly sample up to ``max_frames`` frames into ``out_dir`` as ``{idx}.jpg``
    (sequential 0-based indices). For the Test page's video scrubber — CPU/IO only.
    Returns the number of frames written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:  # some containers don't report a frame count — read sequentially
            written = 0
            while written < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                cv2.imwrite(str(out_dir / f"{written}.jpg"), frame)
                written += 1
            return written
        count = min(max_frames, total)
        step = max(1, total // count)
        written = 0
        for i in range(count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out_dir / f"{written}.jpg"), frame)
            written += 1
        return written
    finally:
        cap.release()


def _append(progress_path: Path, event: dict) -> None:
    with open(progress_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _similarity(a, b) -> float:
    """1.0 == identical, 0.0 == maximally different (mean abs diff on 32x32 gray)."""
    diff = cv2.absdiff(a, b).mean() / 255.0
    return 1.0 - float(diff)


def extract_frames(
    video_path: Path,
    raw_dir: Path,
    progress_path: Path,
    stem: str,
    params: ExtractParams,
    cancel_path: Path,
) -> dict:
    """Extract sampled frames into ``raw_dir``. Returns a summary dict.

    Progress is streamed as JSONL events with a ``phase`` field so the existing
    SSE reader (``app.services.label_manager.read_progress``) can consume it.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    # fresh progress log per run (so resample replays from a clean slate)
    progress_path.write_text("")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _append(progress_path, {"phase": "error", "msg": "Cannot open video"})
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

    _append(
        progress_path,
        {
            "phase": "start",
            "src_fps": round(src_fps, 2),
            "total_frames": total_frames,
            "step": step,
        },
    )

    idx = 0
    saved = 0
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
                    fname = f"{stem}_{saved + 1:05d}.jpg"
                    cv2.imwrite(str(raw_dir / fname), frame)
                    saved += 1
                    if saved % 10 == 0 or saved == 1:
                        _append(
                            progress_path,
                            {
                                "phase": "extract",
                                "saved": saved,
                                "scanned": idx + 1,
                                "total_frames": total_frames,
                                "skipped_dup": skipped_dup,
                            },
                        )
                    if saved >= params.max_frames:
                        break
            idx += 1
    finally:
        cap.release()

    if cancelled:
        _append(progress_path, {"phase": "cancelled", "saved": saved})
        return {"status": "cancelled", "saved": saved, "skipped_dup": skipped_dup}

    _append(
        progress_path,
        {"phase": "done", "saved": saved, "scanned": idx, "skipped_dup": skipped_dup},
    )
    return {"status": "done", "saved": saved, "skipped_dup": skipped_dup, "scanned": idx}

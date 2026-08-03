"""Live-preview detection worker: run YOLO + ByteTrack over a video ONCE, cache the
raw per-sample detections, and transcode a browser-playable H.264 preview. Runs in a
child process (torch/ultralytics + cv2). Progress → jobs_dir/{job_id}/progress.jsonl.

This is the expensive half of the trackcrop pipeline (`detect_video`). The cheap
half (`plan_from_detections`) runs synchronously in the API on every tuning change,
reading the cached detections — so the crop overlay updates instantly without any
re-inference. The frontend plays the preview and draws the crop box on a canvas.

Unlike annotate_worker, the source is NOT rendered here; we keep a playable copy
(preview.mp4) instead of deleting the upload, because the client plays it back.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

# Reuse the annotate worker's H.264 transcode (ffmpeg) — same browser-playability need.
from app.workers.annotate_worker import _to_h264


class _Cancelled(Exception):
    """Raised from the progress callback when a CANCEL sentinel appears."""


def _emit(progress_path: Path, event: dict) -> None:
    with open(progress_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def run_live(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    import cv2

    from app.core.config import resolve_device

    job_dir = Path(jobs_dir) / job_id
    progress = job_dir / "progress.jsonl"
    cancel = job_dir / "CANCEL"

    work = Path(cfg["work"])  # test_dir/live/{job_id} — cache + preview live here
    work.mkdir(parents=True, exist_ok=True)
    src = Path(cfg["source"])
    preview = work / "preview.mp4"

    device = resolve_device(cfg.get("device"))
    conf_cfg = cfg.get("conf")
    conf = float(conf_cfg) if conf_cfg is not None else 0.10  # crop 검출 기본 0.10
    imgsz = int(cfg.get("imgsz", 1920))
    interval = int(cfg.get("sampling_interval_ms") or 100)
    _, pt = cfg["specs"][0]  # detection is single-model — use the first selected model

    try:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required to encode a browser-playable preview but was not "
                "found. Install it (Docker image ships it; locally: `brew install ffmpeg`)."
            )

        # probe geometry / length for progress total + client-side overlay scaling
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration_ms = int(total_frames / fps * 1000) if fps else 0
        total_samples = max(1, duration_ms // interval + 1)

        _emit(progress, {"phase": "start", "total": total_samples})

        # ---- detection (the expensive pass) ----
        # trackcrop pulls in cv2/ultralytics — keep the import lazy (worker only).
        from app.ml.trackcrop.detection import build_detector
        from app.ml.trackcrop.detection_io import dump_detections
        from app.ml.trackcrop.pipeline import detect_video

        entries = cfg.get("detectors") or [
            {"pt": pt, "mode": "full", "conf": conf_cfg, "imgsz": imgsz}
        ]
        detector = build_detector(entries, device, default_conf=conf)

        def on_progress(done: int) -> None:
            if cancel.exists():
                raise _Cancelled()
            if done % 10 == 0 or done >= total_samples:
                _emit(progress, {"phase": "detect", "done": done, "total": total_samples})

        try:
            detected = detect_video(
                src, detector=detector, sampling_interval_ms=interval, on_progress=on_progress
            )
        except _Cancelled:
            _emit(progress, {"phase": "cancelled", "total": total_samples})
            return {"status": "cancelled"}

        (work / "detected.json").write_text(
            json.dumps(dump_detections(detected)), encoding="utf-8"
        )

        # ---- browser-playable preview (transcode; source may be an odd codec) ----
        _emit(progress, {"phase": "encoding", "done": total_samples, "total": total_samples})
        _to_h264(src, preview)

        meta = {
            "source_width": w,
            "source_height": h,
            "fps": fps,
            "duration_ms": duration_ms,
            "sampling_interval_ms": interval,
            "sample_count": len(detected),
        }
        (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        _emit(progress, {"phase": "done", "done": total_samples, "total": total_samples})
        return {"status": "done", "samples": len(detected)}
    except Exception as e:
        _emit(progress, {"phase": "error", "msg": str(e)})
        raise
    finally:
        # keep preview.mp4; the original upload is transient
        src.unlink(missing_ok=True)

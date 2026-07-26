"""Video tracking worker: run a model's object tracker (ByteTrack) over a video,
draw per-object boxes with stable track IDs (and a short motion trail), and write
a browser-playable (H.264) mp4. Runs in a child process (torch/ultralytics + cv2).
Progress → jobs_dir/{job_id}/progress.jsonl.

cv2.VideoWriter H.264 support is unreliable across OpenCV builds, so we always
write an mp4v intermediate then transcode to H.264 with ffmpeg. ffmpeg is REQUIRED
(shipped in the Docker image); if it's missing the job fails loudly rather than
writing an unplayable mp4v file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

# BGR palette (matches the frontend BoxOverlay hue order closely enough).
_PALETTE = [
    (247, 171, 77), (102, 207, 81), (107, 107, 255), (59, 212, 255), (232, 93, 204),
    (151, 201, 32), (43, 146, 255), (172, 131, 247), (252, 143, 116), (75, 227, 169),
]

_TRAIL_LEN = 30  # frames of motion history to draw per track


def _color(key: int) -> tuple[int, int, int]:
    return _PALETTE[key % len(_PALETTE)]


def _emit(progress_path: Path, event: dict) -> None:
    with open(progress_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def run_annotate(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    import cv2

    from app.core.config import resolve_device

    job_dir = Path(jobs_dir) / job_id
    progress = job_dir / "progress.jsonl"
    cancel = job_dir / "CANCEL"

    src = Path(cfg["source"])
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("._raw.mp4")  # mp4v intermediate

    device = resolve_device(cfg.get("device"))
    conf_thr = float(cfg.get("conf", 0.4))
    iou_thr = float(cfg.get("iou", cfg.get("iou_wbf", 0.7)))
    imgsz = int(cfg.get("imgsz", 640))

    try:
        # fail early if we can't produce a browser-playable file
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required to encode a browser-playable video but was not "
                "found. Install it (Docker image ships it; locally: `brew install ffmpeg`)."
            )

        from ultralytics import YOLO

        # tracking is single-model by nature — use the first selected model
        _, pt = cfg["specs"][0]
        model = YOLO(pt)

        # probe geometry / length for the writer + progress total
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        _emit(progress, {"phase": "start", "total": total})

        trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=_TRAIL_LEN))
        idx = 0
        cancelled = False
        results = model.track(
            source=str(src),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_thr,
            iou=iou_thr,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for r in results:
            if cancel.exists():
                cancelled = True
                break
            frame = r.orig_img
            boxes = getattr(r, "boxes", None)
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else [None] * len(xyxy)
                )
                names = r.names
                for k in range(len(xyxy)):
                    x1, y1, x2, y2 = (int(v) for v in xyxy[k])
                    tid = int(ids[k]) if ids[k] is not None else None
                    color = _color(tid if tid is not None else int(clss[k]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cls_name = names.get(int(clss[k]), str(clss[k]))
                    label = (
                        f"#{tid} {cls_name} {confs[k] * 100:.0f}%"
                        if tid is not None
                        else f"{cls_name} {confs[k] * 100:.0f}%"
                    )
                    cv2.putText(
                        frame, label, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                    )
                    if tid is not None:
                        trails[tid].append(((x1 + x2) // 2, (y1 + y2) // 2))
                        pts = trails[tid]
                        for j in range(1, len(pts)):
                            cv2.line(frame, pts[j - 1], pts[j], color, 2, cv2.LINE_AA)
            writer.write(frame)
            idx += 1
            if idx % 5 == 0 or idx == total:
                _emit(progress, {"phase": "annotate", "done": idx, "total": total})

        writer.release()

        if cancelled:
            tmp.unlink(missing_ok=True)
            _emit(progress, {"phase": "cancelled", "done": idx, "total": total})
            return {"status": "cancelled"}

        # transcode mp4v → H.264 for browser <video> playback
        _emit(progress, {"phase": "encoding", "done": idx, "total": total})
        _to_h264(tmp, out)
        tmp.unlink(missing_ok=True)

        _emit(progress, {"phase": "done", "done": idx, "total": total})
        return {"status": "done", "frames": idx}
    except Exception as e:
        _emit(progress, {"phase": "error", "msg": str(e)})
        raise
    finally:
        # source video is transient — never keep it around
        src.unlink(missing_ok=True)


def _to_h264(src: Path, dst: Path) -> None:
    """Transcode to H.264/yuv420p mp4 (browser-safe). ffmpeg is required — the
    caller checks for it up front and fails the job if it's missing."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

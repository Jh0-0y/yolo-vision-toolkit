"""Video annotation worker: draw a model's detections onto every frame and
write a browser-playable (H.264) annotated mp4. Runs in a child process
(torch/ultralytics + cv2). Progress → jobs_dir/{job_id}/progress.jsonl.

cv2.VideoWriter H.264 support is unreliable across OpenCV builds, so we always
write an mp4v intermediate then transcode to H.264 with ffmpeg (verified present).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

# BGR palette (matches the frontend BoxOverlay hue order closely enough).
_PALETTE = [
    (247, 171, 77), (102, 207, 81), (107, 107, 255), (59, 212, 255), (232, 93, 204),
    (151, 201, 32), (43, 146, 255), (172, 131, 247), (252, 143, 116), (75, 227, 169),
]


def _color(cls: int) -> tuple[int, int, int]:
    return _PALETTE[cls % len(_PALETTE)]


def _emit(progress_path: Path, event: dict) -> None:
    with open(progress_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def run_annotate(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    import cv2

    from app.core.config import resolve_device
    from app.ml.predict import PredictConfig, predict_image

    job_dir = Path(jobs_dir) / job_id
    progress = job_dir / "progress.jsonl"
    cancel = job_dir / "CANCEL"

    src = Path(cfg["source"])
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("._raw.mp4")  # mp4v intermediate

    device = resolve_device(cfg.get("device"))
    conf_thr = float(cfg.get("conf", 0.4))

    try:
        from ultralytics import YOLO

        models = []
        for model_id, pt in cfg["specs"]:
            model = YOLO(pt)
            try:
                model.to(device)
            except Exception:
                pass
            models.append((model_id, model))

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        pcfg = PredictConfig(
            conf=conf_thr,
            iou_wbf=float(cfg.get("iou_wbf", 0.55)),
            imgsz=int(cfg.get("imgsz", 640)),
            device=device,
        )
        _emit(progress, {"phase": "start", "total": total})

        idx = 0
        cancelled = False
        while True:
            if cancel.exists():
                cancelled = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            res = predict_image(models, frame, pcfg)
            for b in res["boxes"]:
                if b["score"] < conf_thr:
                    continue
                x1, y1, x2, y2 = b["xyxyn"]
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                color = _color(b["cls"])
                cv2.rectangle(frame, p1, p2, color, 2)
                label = f'{b["name"]} {b["score"] * 100:.0f}%'
                cv2.putText(
                    frame, label, (p1[0], max(12, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
            writer.write(frame)
            idx += 1
            if idx % 5 == 0 or idx == total:
                _emit(progress, {"phase": "annotate", "done": idx, "total": total})

        cap.release()
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
    """Transcode to H.264/yuv420p mp4 (browser-safe). Falls back to a copy if
    ffmpeg is unavailable."""
    if shutil.which("ffmpeg") is None:
        shutil.copyfile(src, dst)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

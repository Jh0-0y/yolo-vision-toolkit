"""Video tracking worker: run a model's object tracker (ByteTrack) over a video,
draw per-object boxes with stable track IDs (and a short motion trail), and/or
apply a moving vertical (9:16) crop window computed by the trackcrop pipeline,
then write a browser-playable (H.264) mp4. Runs in a child process (torch/
ultralytics + cv2). Progress → jobs_dir/{job_id}/progress.jsonl.

Two independent overlays, each toggled by cfg:
  - object_tracking: ByteTrack boxes + IDs + motion trails (per-frame model.track).
  - crop_tracking:   the trackcrop pipeline (100ms predict pass, separate from
                     ByteTrack by design) yields a target-center trajectory. Its
                     crop X coordinates are always written to crop.json. What we do
                     with that trajectory depends on cfg["crop_output"]:
                       "label" — draw the 9:16 crop rectangle onto the full frame
                                 (default; composes with object_tracking boxes).
                       "video" — actually cut each frame down to the vertical 9:16
                                 window and output that clean crop clip (no boxes,
                                 no rectangle; object_tracking is ignored).
    The cut/draw geometry lives in app.domain.crop_render.

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
    object_tracking = bool(cfg.get("object_tracking", True))
    crop_tracking = bool(cfg.get("crop_tracking", True))
    # crop output form: "label" = draw 9:16 rectangle overlay (default),
    # "video" = cut the frame to the vertical window and output that clean clip.
    crop_cut = crop_tracking and cfg.get("crop_output") == "video"

    from app.domain import crop_render  # cv2-only geometry (no torch) — safe here

    # tracking/crop are single-model by nature — use the first selected model
    _, pt = cfg["specs"][0]

    try:
        # fail early if we can't produce a browser-playable file
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required to encode a browser-playable video but was not "
                "found. Install it (Docker image ships it; locally: `brew install ffmpeg`)."
            )

        # probe geometry / length for the writer + progress total
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # crop-cut output is narrower (the vertical 9:16 window); the overlay/box
        # outputs keep the source size.
        crop_w = crop_render.crop_width_for(h, w) if crop_tracking else 0
        out_w, out_h = (crop_w, h) if crop_cut else (w, h)
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

        _emit(progress, {"phase": "start", "total": total})

        # ---- crop trajectory (separate 100ms predict pass) ----
        crop_traj: crop_render.Trajectory | None = None
        if crop_tracking:
            _emit(progress, {"phase": "crop_analyze", "total": total})
            # trackcrop pulls in cv2/ultralytics — keep the import lazy (worker only)
            from app.ml.trackcrop import analyze_video

            # trackcrop is tuned for ball recall (low conf, large imgsz) and its own
            # 1920/608 constants; validate=False so non-1080p input doesn't trip the
            # geometry checks — we only need the target-centre trajectory here.
            cropres = analyze_video(
                str(src), model_path=pt, device=device, imgsz=1280, conf=0.10, validate=False
            )
            crop_traj = crop_render.build_trajectory(cropres.samples, w)
            (out.parent / "crop.json").write_text(cropres.to_json(), encoding="utf-8")

        idx = 0
        cancelled = False

        if crop_cut:
            # crop-cut: output the clean vertical clip — cut each frame to the 9:16
            # window (no boxes, no rectangle; object_tracking is ignored by design).
            cap = cv2.VideoCapture(str(src))
            try:
                while True:
                    if cancel.exists():
                        cancelled = True
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(
                        crop_render.cut_window(frame, idx / fps * 1000.0, crop_traj, crop_w, w)
                    )
                    idx += 1
                    if idx % 5 == 0 or idx == total:
                        _emit(progress, {"phase": "annotate", "done": idx, "total": total})
            finally:
                cap.release()
        elif object_tracking:
            from ultralytics import YOLO

            model = YOLO(pt)
            trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=_TRAIL_LEN))
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
                if crop_traj is not None:
                    crop_render.draw_window(frame, idx / fps * 1000.0, crop_traj, crop_w, w, h)
                writer.write(frame)
                idx += 1
                if idx % 5 == 0 or idx == total:
                    _emit(progress, {"phase": "annotate", "done": idx, "total": total})
        else:
            # crop-only overlay: no ByteTrack pass — just read frames and draw the rectangle
            cap = cv2.VideoCapture(str(src))
            try:
                while True:
                    if cancel.exists():
                        cancelled = True
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if crop_traj is not None:
                        crop_render.draw_window(frame, idx / fps * 1000.0, crop_traj, crop_w, w, h)
                    writer.write(frame)
                    idx += 1
                    if idx % 5 == 0 or idx == total:
                        _emit(progress, {"phase": "annotate", "done": idx, "total": total})
            finally:
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
    """Transcode to H.264/yuv420p mp4 (browser-safe). ffmpeg is required — the
    caller checks for it up front and fails the job if it's missing."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         # yuv420p needs even width/height; pad up by 1px if a crop made a dim odd
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

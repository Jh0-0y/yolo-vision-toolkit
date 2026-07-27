"""Video tracking worker: run a model's object tracker (ByteTrack) over a video,
draw per-object boxes with stable track IDs (and a short motion trail), and/or
overlay a moving vertical (9:16) crop window computed by the trackcrop pipeline,
then write a browser-playable (H.264) mp4. Runs in a child process (torch/
ultralytics + cv2). Progress → jobs_dir/{job_id}/progress.jsonl.

Two independent overlays, each toggled by cfg:
  - object_tracking: ByteTrack boxes + IDs + motion trails (per-frame model.track).
  - crop_tracking:   the trackcrop pipeline (100ms predict pass, separate from
                     ByteTrack by design) yields a target-center trajectory; we
                     draw a vertical 9:16 crop frame that follows it. Its crop X
                     coordinates are also written to crop.json for download.

cv2.VideoWriter H.264 support is unreliable across OpenCV builds, so we always
write an mp4v intermediate then transcode to H.264 with ffmpeg. ffmpeg is REQUIRED
(shipped in the Docker image); if it's missing the job fails loudly rather than
writing an unplayable mp4v file.
"""

from __future__ import annotations

import bisect
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
_CROP_COLOR = (0, 255, 255)  # BGR yellow — distinct, always the crop window


def _color(key: int) -> tuple[int, int, int]:
    return _PALETTE[key % len(_PALETTE)]


def _emit(progress_path: Path, event: dict) -> None:
    with open(progress_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def _build_crop_traj(samples: list, width: int) -> tuple[list[int], list[float]]:
    """trackcrop samples → (ms_list, center_x_list) for per-frame interpolation.

    The 'center' fallback stores 1920/2=960 (trackcrop's fixed source width); remap
    it to the actual frame centre so the overlay is correct at any resolution.
    """
    ms_list: list[int] = []
    cx_list: list[float] = []
    half = width / 2
    for s in samples:
        ms_list.append(s.video_offset_ms)
        cx_list.append(half if s.target_type == "center" else s.target_center_x)
    return ms_list, cx_list


def _crop_center_at(ms: float, traj: tuple[list[int], list[float]]) -> float | None:
    """Linear-interpolate the target centre X at time `ms` from the 100ms grid."""
    ms_list, cx_list = traj
    if not ms_list:
        return None
    if ms <= ms_list[0]:
        return cx_list[0]
    if ms >= ms_list[-1]:
        return cx_list[-1]
    i = bisect.bisect_right(ms_list, ms)  # ms_list[i-1] <= ms < ms_list[i]
    t0, t1 = ms_list[i - 1], ms_list[i]
    x0, x1 = cx_list[i - 1], cx_list[i]
    if t1 == t0:
        return x0
    r = (ms - t0) / (t1 - t0)
    return x0 + (x1 - x0) * r


def _draw_crop_window(
    frame, ms: float, traj: tuple[list[int], list[float]], crop_w: int, width: int, height: int
) -> None:
    """Draw the full-height vertical crop frame centred on the interpolated target."""
    import cv2

    cx = _crop_center_at(ms, traj)
    if cx is None:
        return
    left = int(round(cx - crop_w / 2))
    left = max(0, min(left, width - crop_w))
    right = left + crop_w
    cv2.rectangle(frame, (left, 0), (right - 1, height - 1), _CROP_COLOR, 3)
    cv2.putText(
        frame, "CROP", (left + 6, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CROP_COLOR, 2, cv2.LINE_AA,
    )


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
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        _emit(progress, {"phase": "start", "total": total})

        # ---- crop trajectory (separate 100ms predict pass) ----
        crop_traj: tuple[list[int], list[float]] | None = None
        crop_w = 0
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
            crop_traj = _build_crop_traj(cropres.samples, w)
            crop_w = max(2, round(h * 9 / 16))
            (out.parent / "crop.json").write_text(cropres.to_json(), encoding="utf-8")

        idx = 0
        cancelled = False

        if object_tracking:
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
                if crop_tracking and crop_traj is not None:
                    _draw_crop_window(frame, idx / fps * 1000.0, crop_traj, crop_w, w, h)
                writer.write(frame)
                idx += 1
                if idx % 5 == 0 or idx == total:
                    _emit(progress, {"phase": "annotate", "done": idx, "total": total})
        else:
            # crop-only: no ByteTrack pass — just read frames and draw the crop window
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
                        _draw_crop_window(frame, idx / fps * 1000.0, crop_traj, crop_w, w, h)
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
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

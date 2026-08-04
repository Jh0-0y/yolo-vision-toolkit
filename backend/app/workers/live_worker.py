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


# BGR — 캐시된 검출 박스 오버레이용 (선수: track_id 팔레트, 공: 주황)
_BOX_PALETTE = [
    (247, 171, 77), (102, 207, 81), (107, 107, 255), (59, 212, 255), (232, 93, 204),
    (151, 201, 32), (43, 146, 255), (172, 131, 247), (252, 143, 116), (75, 227, 169),
]
_BALL_COLOR = (0, 140, 255)


def run_live_render(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    """캐시된 검출 + 현재 튜닝으로 오버레이 영상을 렌더한다 — 추론 없음.

    Draw 탭의 캔버스 오버레이(검출 박스·크롭 박스·데드존·중심선·하이라이트)를
    preview.mp4 위에 구워 work/render.mp4 로 저장한다. 검출 캐시를 재사용하므로
    모델·GPU가 필요 없고, 튜닝만 바뀐 재렌더가 빠르다.
    """
    import bisect
    import json as _json

    import cv2

    from app.domain import crop_render
    from app.ml.trackcrop.balltrack import resolve_clip_config
    from app.ml.trackcrop.detection_io import load_detections
    from app.ml.trackcrop.pipeline import plan_from_detections
    from app.workers.annotate_worker import _to_h264

    job_dir = Path(jobs_dir) / job_id
    progress = job_dir / "progress.jsonl"
    cancel = job_dir / "CANCEL"

    work = Path(cfg["work"])  # live 세션 dir — detected.json/preview.mp4 위치
    src = work / "preview.mp4"
    out = work / "render.mp4"
    tmp = work / "._render_raw.mp4"
    overrides = cfg.get("overrides") or {}
    toggles = cfg.get("toggles") or {}
    show_boxes = bool(toggles.get("obj_boxes", True))
    draw_crop_box = bool(toggles.get("draw_crop_box", True))
    show_dead_zone = bool(toggles.get("show_dead_zone", True))
    show_center_line = bool(toggles.get("show_center_line", True))
    show_highlight = bool(toggles.get("show_highlight", True))

    try:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to encode the rendered video.")
        if not src.exists():
            raise RuntimeError("Preview video not found — run detection first.")

        detected = load_detections(
            _json.loads((work / "detected.json").read_text(encoding="utf-8"))
        )
        cropres = plan_from_detections(
            detected, overrides=overrides, collect_debug=show_highlight, validate=False
        )

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 라이브 캐시는 원본 해상도(1920 기준) 좌표 — 프리뷰 해상도에 맞게 스케일
        meta = _json.loads((work / "meta.json").read_text(encoding="utf-8"))
        scale = w / float(meta.get("source_width") or w)

        traj = crop_render.build_trajectory(cropres.samples, int(w / scale))
        traj = (traj[0], [x * scale for x in traj[1]])
        types = crop_render.build_types(cropres.samples)
        debug_lookup = (
            crop_render.build_debug_lookup(cropres.debug) if cropres.debug else None
        )
        crop_w = crop_render.crop_width_for(h, w)
        cfg_resolved = resolve_clip_config(overrides)
        dead_zone_half = cfg_resolved.dead_zone_half * scale

        det_ms = [ms for ms, _ in detected]

        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        _emit(progress, {"phase": "start", "total": total})

        idx = 0
        cancelled = False
        try:
            while True:
                if cancel.exists():
                    cancelled = True
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                ms = idx / fps * 1000.0

                if show_boxes and det_ms:
                    i = min(
                        max(bisect.bisect_right(det_ms, ms) - 1, 0), len(detected) - 1
                    )
                    for d in detected[i][1]:
                        x1 = int(d.bbox_x * scale)
                        y1 = int(d.bbox_y * scale)
                        x2 = int((d.bbox_x + d.bbox_width) * scale)
                        y2 = int((d.bbox_y + d.bbox_height) * scale)
                        if d.object_type == "ball":
                            color = _BALL_COLOR
                            label = f"ball {d.confidence:.0%}"
                        else:
                            color = _BOX_PALETTE[(d.track_id or 0) % len(_BOX_PALETTE)]
                            label = f"#{d.track_id} {d.confidence:.0%}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame, label, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                        )

                if draw_crop_box:
                    crop_render.draw_window(frame, ms, traj, crop_w, w, h)
                    crop_render.draw_target_overlay(
                        frame, ms, traj, types, dead_zone_half, w, h,
                        show_dead_zone=show_dead_zone,
                        show_center_line=show_center_line,
                    )
                if debug_lookup is not None:
                    scaled = crop_render.build_debug_lookup(
                        [
                            {
                                "video_offset_ms": e["video_offset_ms"],
                                "ball_bbox": [v * scale for v in e["ball_bbox"]] if e.get("ball_bbox") else None,
                                "carrier_bbox": [v * scale for v in e["carrier_bbox"]] if e.get("carrier_bbox") else None,
                            }
                            for e in cropres.debug
                        ]
                    ) if abs(scale - 1.0) > 1e-6 else debug_lookup
                    crop_render.draw_selection_overlay(frame, ms, scaled, w, h)
                    debug_lookup = scaled  # 스케일 재계산 1회로 제한

                writer.write(frame)
                idx += 1
                if idx % 10 == 0 or idx == total:
                    _emit(progress, {"phase": "render", "done": idx, "total": total})
        finally:
            cap.release()
            writer.release()

        if cancelled:
            tmp.unlink(missing_ok=True)
            _emit(progress, {"phase": "cancelled", "done": idx, "total": total})
            return {"status": "cancelled"}

        _emit(progress, {"phase": "encoding", "done": idx, "total": total})
        _to_h264(tmp, out)
        tmp.unlink(missing_ok=True)
        _emit(progress, {"phase": "done", "done": idx, "total": total})
        return {"status": "done", "frames": idx}
    except Exception as e:
        _emit(progress, {"phase": "error", "msg": str(e)})
        raise

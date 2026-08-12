"""Video tracking worker: run a model's object tracker (ByteTrack) over a video,
draw per-object boxes with stable track IDs (and a short motion trail), and/or
apply a moving vertical (9:16) crop window computed by the adaptive-crop pipeline,
then write a browser-playable (H.264) mp4. Runs in a child process (torch/
ultralytics + cv2). Progress → jobs_dir/{job_id}/progress.jsonl.

Two independent overlays, each toggled by cfg:
  - object_tracking: ByteTrack boxes + IDs + motion trails (per-frame model.track).
  - crop_tracking:   the adaptive-crop pipeline (100ms predict pass, separate from
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
from collections import defaultdict, deque
from pathlib import Path

from infra import jobs
from lib import video

# BGR palette (matches the frontend BoxOverlay hue order closely enough).
_PALETTE = [
    (247, 171, 77), (102, 207, 81), (107, 107, 255), (59, 212, 255), (232, 93, 204),
    (151, 201, 32), (43, 146, 255), (172, 131, 247), (252, 143, 116), (75, 227, 169),
]

_TRAIL_LEN = 30  # frames of motion history to draw per track


def _color(key: int) -> tuple[int, int, int]:
    return _PALETTE[key % len(_PALETTE)]


def run_annotate(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    import cv2

    from app.core.config import resolve_device

    job = jobs.at(Path(jobs_dir), job_id)
    progress = job.progress_path

    src = Path(cfg["source"])
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("._raw.mp4")  # mp4v intermediate

    device = resolve_device(cfg.get("device"))
    # conf None → 파이프라인별 기본값 (object tracking 0.4 / crop 검출 0.10)
    conf_cfg = cfg.get("conf")
    conf_thr = float(conf_cfg) if conf_cfg is not None else 0.4
    crop_conf = float(conf_cfg) if conf_cfg is not None else 0.10
    iou_thr = float(cfg.get("iou", cfg.get("iou_wbf", 0.7)))
    # 추론 입력 사이즈 — 보통 학습 사이즈에 맞춘다 (UI에서 선택)
    imgsz = int(cfg.get("imgsz", 1280))
    object_tracking = bool(cfg.get("object_tracking", True))
    crop_tracking = bool(cfg.get("crop_tracking", True))
    # crop output form: "none" = JSON만(영상 렌더 스킵), "label" = 9:16 사각형 오버레이,
    # "video" = 프레임을 세로 창으로 잘라 깨끗한 클립 출력.
    crop_output = cfg.get("crop_output", "label")
    crop_cut = crop_tracking and crop_output == "video"
    crop_json_only = crop_tracking and crop_output == "none"
    # 그리기 도구 오버레이 개별 토글 (기본 True = 과거 동작 보존)
    draw_crop_box = bool(cfg.get("draw_crop_box", True))
    show_dead_zone = bool(cfg.get("show_dead_zone", True))
    show_center_line = bool(cfg.get("show_center_line", True))
    show_target_highlight = bool(cfg.get("show_target_highlight", False))
    overrides = cfg.get("overrides") or None  # 크롭 튜닝 오버라이드 (dict)

    from app.domain import crop_render  # cv2-only geometry (no torch) — safe here

    # crop_source: 추론 없이 컷만 하는 모드 ("json" = 업로드 좌표 따라, "center" = 중앙 고정)
    crop_source = cfg.get("crop_source")

    try:
        if crop_source in ("json", "center"):
            return _run_crop_cut(cfg, src, out, tmp, job, crop_source)

        # tracking/crop are single-model by nature — use the first selected model
        _, pt = cfg["specs"][0]

        # fail early if we can't produce a browser-playable file (JSON-only needs no encode)
        if not crop_json_only:
            video.require_ffmpeg()

        # probe geometry / length for the writer + progress total
        meta = video.probe(src)
        fps, total, w, h = meta.fps, meta.frame_count, meta.width, meta.height

        # crop-cut output is narrower (the vertical 9:16 window); the overlay/box
        # outputs keep the source size.
        crop_w = crop_render.crop_width_for(h, w) if crop_tracking else 0
        out_w, out_h = (crop_w, h) if crop_cut else (w, h)
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

        jobs.emit(progress, {"phase": "start", "total": total})

        # ---- crop trajectory (separate predict pass) ----
        crop_traj: crop_render.Trajectory | None = None
        crop_types: tuple[list, list] | None = None  # 디버그 HUD 타입 라벨용
        crop_debug: tuple[list, list] | None = None  # 대상 하이라이트 조회 lookup
        dead_zone_half: float | None = None  # 데드존 밴드 반폭 (crop_tracking일 때만)
        if crop_tracking:
            jobs.emit(progress, {"phase": "crop_analyze", "total": total})
            # adaptive_crop pulls in cv2/ultralytics — keep the imports lazy (worker only)
            from adaptive_crop import (
                VideoInfo,
                build_detector,
                detect_video,
                plan_from_detections,
            )

            from app.ml import crop as crop_adapter

            # 검출은 공 재현율에 맞춘 설정(낮은 conf, 큰 imgsz)으로 따로 한 패스 돈다.
            # 영상 규격은 위에서 이미 읽었으므로 다시 열지 않고 그대로 넘긴다(출처 하나).
            entries = cfg.get("detectors") or [
                {"pt": pt, "mode": "full", "conf": conf_cfg, "imgsz": imgsz}
            ]
            crop_detector = build_detector(
                crop_adapter.detector_entries(entries, crop_conf), device
            )
            crop_cfg = crop_adapter.resolve_clip_config(overrides)
            detected = detect_video(
                str(src),
                detector=crop_detector,
                sampling_interval_ms=crop_cfg.sampling_interval_ms,
            )
            # 컨테이너가 프레임 수를 안 들고 있으면(total=0) 샘플 격자의 끝을 길이로 본다.
            duration_ms = int(total / fps * 1000) if total > 0 and fps else 0
            if duration_ms <= 0 and detected:
                duration_ms = detected[-1][0]
            vinfo = VideoInfo(width=w, height=h, fps=fps, duration_ms=duration_ms)
            cropres = plan_from_detections(
                detected,
                crop_adapter.crop_spec_for(w, h),
                vinfo,
                config=crop_cfg,
                collect_debug=show_target_highlight and not crop_json_only,
            )
            crop_traj = crop_render.build_trajectory(cropres.samples, w)
            crop_types = crop_render.build_types(cropres.samples)
            if cropres.debug:
                crop_debug = crop_render.build_debug_lookup(cropres.debug)
            dead_zone_half = crop_cfg.dead_zone_half
            (out.parent / "crop.json").write_text(
                crop_adapter.crop_plan_json(cropres), encoding="utf-8"
            )

        # ---- JSON-only: crop.json은 위에서 썼다. 영상 렌더/인코딩 스킵 ----
        if crop_json_only:
            writer.release()
            tmp.unlink(missing_ok=True)
            jobs.emit(progress, {"phase": "done", "done": 0, "total": total})
            return {"status": "done", "frames": 0, "json_only": True}

        def _apply_crop_overlays(frame, ms: float) -> None:
            """그리기 토글에 따라 크롭 박스(+데드존·센터선)·대상 하이라이트를 그린다."""
            if crop_traj is None:
                return
            if draw_crop_box:
                crop_render.draw_window(frame, ms, crop_traj, crop_w, w, h)
                crop_render.draw_target_overlay(
                    frame, ms, crop_traj, crop_types, dead_zone_half, w, h,
                    show_dead_zone=show_dead_zone, show_center_line=show_center_line,
                )
            if crop_debug is not None:  # 대상 하이라이트 (독립 토글)
                crop_render.draw_selection_overlay(frame, ms, crop_debug, w, h)

        idx = 0
        cancelled = False

        if crop_cut:
            # crop-cut: output the clean vertical clip — cut each frame to the 9:16
            # window (no boxes, no rectangle; object_tracking is ignored by design).
            cap = cv2.VideoCapture(str(src))
            try:
                while True:
                    if job.cancelled():
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
                        jobs.emit(progress, {"phase": "annotate", "done": idx, "total": total})
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
                if job.cancelled():
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
                    _apply_crop_overlays(frame, idx / fps * 1000.0)
                writer.write(frame)
                idx += 1
                if idx % 5 == 0 or idx == total:
                    jobs.emit(progress, {"phase": "annotate", "done": idx, "total": total})
        else:
            # crop-only overlay: no ByteTrack pass — just read frames and draw the rectangle
            cap = cv2.VideoCapture(str(src))
            try:
                while True:
                    if job.cancelled():
                        cancelled = True
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if crop_traj is not None:
                        _apply_crop_overlays(frame, idx / fps * 1000.0)
                    writer.write(frame)
                    idx += 1
                    if idx % 5 == 0 or idx == total:
                        jobs.emit(progress, {"phase": "annotate", "done": idx, "total": total})
            finally:
                cap.release()

        writer.release()

        if cancelled:
            tmp.unlink(missing_ok=True)
            jobs.emit(progress, {"phase": "cancelled", "done": idx, "total": total})
            return {"status": "cancelled"}

        # transcode mp4v → H.264 for browser <video> playback
        jobs.emit(progress, {"phase": "encoding", "done": idx, "total": total})
        video.to_h264(tmp, out)
        tmp.unlink(missing_ok=True)

        jobs.emit(progress, {"phase": "done", "done": idx, "total": total})
        return {"status": "done", "frames": idx}
    except Exception as e:
        jobs.emit(progress, {"phase": "error", "msg": str(e)})
        raise
    finally:
        # source video is transient — never keep it around
        src.unlink(missing_ok=True)


def _run_crop_cut(cfg, src, out, tmp, job, mode: str) -> dict:
    """추론 없이 세로 9:16 크롭 클립만 만든다.

    mode="json":   업로드된 crop.json의 samples 좌표를 따라 컷.
    mode="center": crop.json 없이 화면 중앙에 고정해 컷.
    모델을 로드하지 않으므로 빠르다.
    """
    import cv2

    from app.domain import crop_render

    progress = job.progress_path

    video.require_ffmpeg()

    meta = video.probe(src)
    fps, total, w, h = meta.fps, meta.frame_count, meta.width, meta.height

    crop_w = crop_render.crop_width_for(h, w)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, h))
    jobs.emit(progress, {"phase": "start", "total": total})

    # 궤적: json → 좌표 따라 / center → 빈 궤적(cut_window이 중앙 fallback)
    traj: crop_render.Trajectory = ([], [])
    if mode == "json":
        data = json.loads(Path(cfg["crop_json_path"]).read_text(encoding="utf-8"))
        keyframes = data.get("keyframes") or []
        if keyframes and "videoOffsetMs" in keyframes[0]:
            # 계약 스키마(camelCase): keyframes의 x는 크롭 왼쪽 X — 중심으로 변환.
            # LINEAR 보간 계약은 cut_window의 np.interp가 그대로 충족한다.
            spec_src_w = float((data.get("source") or {}).get("width") or w)
            spec_crop_w = float((data.get("crop") or {}).get("width") or crop_w)
            scale = w / spec_src_w if spec_src_w else 1.0  # 실제 해상도가 다르면 비례
            ms_list = [int(k["videoOffsetMs"]) for k in keyframes]
            cx_list = [(float(k["x"]) + spec_crop_w / 2) * scale for k in keyframes]
            traj = (ms_list, cx_list)
        else:
            # 레거시 형식: samples(target_center_x) 기반
            samples = data.get("samples") or []
            half = w / 2
            ms_list = [int(s["video_offset_ms"]) for s in samples]
            cx_list = [
                half if s.get("target_type") == "center" else float(s["target_center_x"])
                for s in samples
            ]
            traj = (ms_list, cx_list)

    idx = 0
    cancelled = False
    cap = cv2.VideoCapture(str(src))
    try:
        while True:
            if job.cancelled():
                cancelled = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(crop_render.cut_window(frame, idx / fps * 1000.0, traj, crop_w, w))
            idx += 1
            if idx % 5 == 0 or idx == total:
                jobs.emit(progress, {"phase": "annotate", "done": idx, "total": total})
    finally:
        cap.release()
    writer.release()

    if cancelled:
        tmp.unlink(missing_ok=True)
        jobs.emit(progress, {"phase": "cancelled", "done": idx, "total": total})
        return {"status": "cancelled"}

    jobs.emit(progress, {"phase": "encoding", "done": idx, "total": total})
    video.to_h264(tmp, out)
    tmp.unlink(missing_ok=True)
    jobs.emit(progress, {"phase": "done", "done": idx, "total": total})
    return {"status": "done", "frames": idx}

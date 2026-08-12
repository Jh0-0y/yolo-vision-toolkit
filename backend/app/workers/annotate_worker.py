"""영상 주석 워커 — 클립 하나를 받아 오버레이 영상이나 세로 크롭 클립을 만든다.

자식 프로세스에서 돈다(torch·ultralytics·cv2). 진행률은 `jobs_dir/{job_id}/progress.jsonl`.

설정 조합에 따라 만드는 것이 달라지지만 **골격은 하나다**:

    궤적을 구한다  →  프레임 소스를 고른다  →  스테이지를 쌓는다  →  한 루프로 돌린다

  궤적(crop.Trajectory)   검출 패스로 계산하거나(crop_tracking), 업로드된 crop.json 을
                          따르거나(crop_source), 아예 없다(객체 추적만).
  프레임 소스             영상을 그대로 읽거나, ByteTrack 을 돌려 검출과 함께 읽는다.
  스테이지                프레임에 무엇을 그릴지 — 객체 박스 · 크롭 창 · HUD · 하이라이트.
  출력 변환               원본 크기 그대로 두거나, 세로 창으로 잘라낸다.

무엇을 켤지는 `_plan_render` 한 곳에서만 정한다. 새 오버레이를 붙이려면 스테이지를
하나 만들어 거기에 한 줄 더하면 되고, 루프는 건드리지 않는다.

cv2.VideoWriter 의 H.264 지원은 OpenCV 빌드마다 달라 믿을 수 없다. 그래서 항상 mp4v
중간 파일을 쓰고 ffmpeg 로 다시 인코딩한다 — ffmpeg 는 필수이며, 없으면 재생 불가능한
파일을 남기는 대신 잡을 실패시킨다.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

from infra import jobs
from lib import crop, video

# BGR palette (matches the frontend BoxOverlay hue order closely enough).
_PALETTE = [
    (247, 171, 77), (102, 207, 81), (107, 107, 255), (59, 212, 255), (232, 93, 204),
    (151, 201, 32), (43, 146, 255), (172, 131, 247), (252, 143, 116), (75, 227, 169),
]

_TRAIL_LEN = 30  # frames of motion history to draw per track


def _color(key: int) -> tuple[int, int, int]:
    return _PALETTE[key % len(_PALETTE)]


# ---------- 궤적: "이 시각에 어디를 잘라야 하나" ----------


def _trajectory_from_json(path: str, frame_width: int, crop_w: int) -> crop.Trajectory:
    """업로드된 crop.json 을 궤적으로 읽는다. 두 형식을 모두 받는다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    keyframes = data.get("keyframes") or []
    if keyframes and "videoOffsetMs" in keyframes[0]:
        # 계약 스키마(camelCase): keyframes의 x는 크롭 왼쪽 X — 중심으로 변환.
        # LINEAR 보간 계약은 center_at 의 선형 보간이 그대로 충족한다.
        spec_src_w = float((data.get("source") or {}).get("width") or frame_width)
        spec_crop_w = float((data.get("crop") or {}).get("width") or crop_w)
        scale = frame_width / spec_src_w if spec_src_w else 1.0  # 해상도가 다르면 비례
        return (
            [int(k["videoOffsetMs"]) for k in keyframes],
            [(float(k["x"]) + spec_crop_w / 2) * scale for k in keyframes],
        )
    # 레거시 형식: samples(target_center_x) 기반
    samples = data.get("samples") or []
    half = frame_width / 2
    return (
        [int(s["video_offset_ms"]) for s in samples],
        [
            half if s.get("target_type") == "center" else float(s["target_center_x"])
            for s in samples
        ],
    )


def _detect_trajectory(cfg, src: Path, meta, device: str, *, collect_debug: bool):
    """검출 패스를 한 번 돌려 크롭 궤적을 계산하고 crop.json 을 쓴다.

    ByteTrack(객체 추적)과는 **일부러 분리된 패스**다 — 공 재현율에 맞춘 설정(낮은
    conf, 큰 imgsz)으로 따로 돌아야 크롭 좌표가 안정적이다.

    돌려주는 것: (궤적, 타입 조회, 디버그 조회, 데드존 반폭, crop.json 문자열).
    """
    # adaptive_crop pulls in cv2/ultralytics — keep the imports lazy (worker only)
    from adaptive_crop import VideoInfo, build_detector, detect_video, plan_from_detections

    from lib.crop import plan as crop_adapter

    conf_cfg = cfg.get("conf")
    crop_conf = float(conf_cfg) if conf_cfg is not None else 0.10
    entries = cfg.get("detectors") or [
        {"pt": cfg["specs"][0][1], "mode": "full", "conf": conf_cfg,
         "imgsz": int(cfg.get("imgsz", 1280))}
    ]
    detector = build_detector(crop_adapter.detector_entries(entries, crop_conf), device)
    clip_cfg = crop_adapter.resolve_clip_config(cfg.get("overrides") or None)

    detected = detect_video(
        str(src), detector=detector, sampling_interval_ms=clip_cfg.sampling_interval_ms
    )
    # 컨테이너가 프레임 수를 안 들고 있으면(0) 샘플 격자의 끝을 길이로 본다.
    duration_ms = meta.duration_ms
    if duration_ms <= 0 and detected:
        duration_ms = detected[-1][0]
    vinfo = VideoInfo(
        width=meta.width, height=meta.height, fps=meta.fps, duration_ms=duration_ms
    )
    result = plan_from_detections(
        detected,
        crop_adapter.crop_spec_for(meta.width, meta.height),
        vinfo,
        config=clip_cfg,
        collect_debug=collect_debug,
    )
    return (
        crop.geometry.build_trajectory(result.samples, meta.width),
        crop.geometry.build_types(result.samples),
        crop.geometry.build_debug_lookup(result.debug) if result.debug else None,
        clip_cfg.dead_zone_half,
        crop_adapter.crop_plan_json(result),
    )


# ---------- 프레임 소스: (idx, frame, ms, 검출) 을 낸다 ----------


def _decoded_frames(src: Path, fps: float):
    """영상을 순서대로 읽는다. 검출은 없다."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield idx, frame, idx / fps * 1000.0, None
            idx += 1
    finally:
        cap.release()


def _tracked_frames(pt: str, src: Path, fps: float, *, conf, iou, imgsz, device):
    """ByteTrack 을 돌려 프레임과 그 프레임의 검출을 함께 낸다."""
    from ultralytics import YOLO

    results = YOLO(pt).track(
        source=str(src),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    for idx, r in enumerate(results):
        yield idx, r.orig_img, idx / fps * 1000.0, r


# ---------- 스테이지: 프레임에 무엇을 그릴지 ----------


def _box_stage():
    """검출 박스 · 라벨 · 최근 이동 궤적. 트랙 id 별 궤적을 프레임 간에 기억한다."""
    import cv2

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=_TRAIL_LEN))

    def draw(frame, ms: float, detected) -> None:
        boxes = getattr(detected, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [None] * len(xyxy)
        )
        names = detected.names
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

    return draw


def _crop_overlay_stage(cfg, traj, types, debug, dead_zone_half, crop_w, w, h):
    """크롭 창 사각형 · HUD(데드존·중심선·타입) · 선택 하이라이트."""
    draw_crop_box = bool(cfg.get("draw_crop_box", True))
    show_dead_zone = bool(cfg.get("show_dead_zone", True))
    show_center_line = bool(cfg.get("show_center_line", True))

    def draw(frame, ms: float, _detected) -> None:
        cx = crop.geometry.center_at(ms, traj)  # 보간은 프레임당 한 번
        if draw_crop_box and cx is not None:
            crop.window.draw(frame, cx, crop_w, w, h)
            crop.hud.draw(
                frame, cx, w, h,
                target_type=crop.geometry.type_at(ms, types) if types else None,
                dead_zone_half=dead_zone_half,
                show_dead_zone=show_dead_zone, show_center_line=show_center_line,
            )
        if debug is not None:  # 대상 하이라이트 (독립 토글)
            crop.highlight.draw(frame, crop.geometry.debug_at(ms, debug), w, h)

    return draw


def run_annotate(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    import cv2

    from app.core.config import resolve_device

    job = jobs.at(Path(jobs_dir), job_id)
    progress = job.progress_path

    src = Path(cfg["source"])
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("._raw.mp4")  # mp4v intermediate

    # crop_source: 추론 없이 컷만 하는 모드 ("json" = 업로드 좌표 따라, "center" = 중앙 고정)
    crop_source = cfg.get("crop_source")
    crop_tracking = crop_source is None and bool(cfg.get("crop_tracking", True))
    # crop_output: "none" = JSON만(영상 렌더 스킵) · "label" = 오버레이 · "video" = 세로 컷
    crop_output = cfg.get("crop_output", "label")
    json_only = crop_tracking and crop_output == "none"
    # 컷 모드는 깨끗한 세로 클립만 낸다 — 오버레이도 객체 추적도 무시한다(설계).
    cut_output = crop_source is not None or (crop_tracking and crop_output == "video")
    object_tracking = (
        crop_source is None and not cut_output and bool(cfg.get("object_tracking", True))
    )

    try:
        if not json_only:
            video.require_ffmpeg()  # 한 시간 렌더한 뒤 마지막에 실패하지 않도록 먼저

        meta = video.probe(src)
        fps, total, w, h = meta.fps, meta.frame_count, meta.width, meta.height
        crop_w = crop.geometry.crop_width_for(h, w) if (crop_tracking or crop_source) else 0

        jobs.emit(progress, {"phase": "start", "total": total})

        # ---- 궤적 ----
        traj = types = debug = dead_zone_half = None
        if crop_source is not None:
            # json → 업로드 좌표 / center → 빈 궤적(cut 이 중앙으로 폴백)
            traj = (
                _trajectory_from_json(cfg["crop_json_path"], w, crop_w)
                if crop_source == "json"
                else ([], [])
            )
        elif crop_tracking:
            jobs.emit(progress, {"phase": "crop_analyze", "total": total})
            traj, types, debug, dead_zone_half, plan_json = _detect_trajectory(
                cfg, src, meta, resolve_device(cfg.get("device")),
                collect_debug=bool(cfg.get("show_target_highlight", False)) and not json_only,
            )
            (out.parent / "crop.json").write_text(plan_json, encoding="utf-8")

        # ---- JSON만: 좌표는 위에서 썼다. 렌더·인코딩은 건너뛴다 ----
        if json_only:
            jobs.emit(progress, {"phase": "done", "done": 0, "total": total})
            return {"status": "done", "frames": 0, "json_only": True}

        # ---- 무엇을 켤지 정하는 유일한 곳 ----
        if object_tracking:
            conf_cfg = cfg.get("conf")
            source = _tracked_frames(
                cfg["specs"][0][1], src, fps,
                conf=float(conf_cfg) if conf_cfg is not None else 0.4,
                iou=float(cfg.get("iou", cfg.get("iou_wbf", 0.7))),
                imgsz=int(cfg.get("imgsz", 1280)),
                device=resolve_device(cfg.get("device")),
            )
        else:
            source = _decoded_frames(src, fps)

        stages = []
        if object_tracking:
            stages.append(_box_stage())
        if traj is not None and not cut_output:
            stages.append(
                _crop_overlay_stage(cfg, traj, types, debug, dead_zone_half, crop_w, w, h)
            )

        out_w, out_h = (crop_w, h) if cut_output else (w, h)
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

        # ---- 단일 렌더 루프 ----
        idx = 0
        cancelled = False
        for i, frame, ms, detected in source:
            if job.cancelled():
                cancelled = True
                break
            for stage in stages:
                stage(frame, ms, detected)
            if cut_output:
                frame = crop.cut.window(
                    frame, crop.geometry.center_at(ms, traj), crop_w, w
                )
            writer.write(frame)
            idx = i + 1
            if idx % 5 == 0 or idx == total:
                jobs.emit(progress, {"phase": "annotate", "done": idx, "total": total})
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

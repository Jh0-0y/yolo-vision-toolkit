"""Live-preview detection worker: run YOLO + ByteTrack over a video ONCE, cache the
raw per-sample detections, and transcode a browser-playable H.264 preview. Runs in a
child process (torch/ultralytics + cv2). Progress → jobs_dir/{job_id}/progress.jsonl.

This is the expensive half of the adaptive-crop pipeline (`detect_video`). The cheap
half (`plan_from_detections`) runs synchronously in the API on every tuning change,
reading the cached detections — so the crop overlay updates instantly without any
re-inference. The frontend plays the preview and draws the crop box on a canvas.

Unlike annotate_worker, the source is NOT rendered here; we keep a playable copy
(preview.mp4) instead of deleting the upload, because the client plays it back.
"""

from __future__ import annotations

import json
from pathlib import Path

from infra import jobs
from lib import crop, video


class _Cancelled(Exception):
    """Raised from the progress callback when a CANCEL sentinel appears."""


def run_live(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    from app.core.config import resolve_device

    job = jobs.at(Path(jobs_dir), job_id)
    progress = job.progress_path

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
        video.require_ffmpeg()

        # adaptive_crop pulls in cv2/ultralytics — keep the imports lazy (worker only).
        from adaptive_crop import build_detector, detect_video, probe_video
        from adaptive_crop.detect.io import dump_detections

        from lib.crop import plan as crop_adapter

        # probe geometry / length for progress total + client-side overlay scaling
        vinfo = probe_video(src)
        w, h, fps, duration_ms = vinfo.width, vinfo.height, vinfo.fps, vinfo.duration_ms
        total_samples = max(1, duration_ms // interval + 1)

        jobs.emit(progress, {"phase": "start", "total": total_samples})

        # ---- detection (the expensive pass) ----
        entries = cfg.get("detectors") or [
            {"pt": pt, "mode": "full", "conf": conf_cfg, "imgsz": imgsz}
        ]
        detector = build_detector(crop_adapter.detector_entries(entries, conf), device)

        def on_progress(done: int) -> None:
            if job.cancelled():
                raise _Cancelled()
            if done % 10 == 0 or done >= total_samples:
                jobs.emit(progress, {"phase": "detect", "done": done, "total": total_samples})

        try:
            detected = detect_video(
                src, detector=detector, sampling_interval_ms=interval, on_progress=on_progress
            )
        except _Cancelled:
            jobs.emit(progress, {"phase": "cancelled", "total": total_samples})
            return {"status": "cancelled"}

        (work / "detected.json").write_text(
            json.dumps(dump_detections(detected)), encoding="utf-8"
        )

        # ---- browser-playable preview (transcode; source may be an odd codec) ----
        jobs.emit(progress, {"phase": "encoding", "done": total_samples, "total": total_samples})
        video.to_h264(src, preview)

        meta = {
            "source_width": w,
            "source_height": h,
            "fps": fps,
            # 컨테이너가 프레임 수를 안 들고 있으면(0) 샘플 격자의 끝을 길이로 본다 —
            # 좌표 계산이 이 값을 클립 길이로 쓴다.
            "duration_ms": duration_ms or (detected[-1][0] if detected else 0),
            "sampling_interval_ms": interval,
            "sample_count": len(detected),
        }
        (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        jobs.emit(progress, {"phase": "done", "done": total_samples, "total": total_samples})
        return {"status": "done", "samples": len(detected)}
    except Exception as e:
        jobs.emit(progress, {"phase": "error", "msg": str(e)})
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
    from adaptive_crop import plan_from_detections
    from adaptive_crop.detect.io import load_detections

    from lib.crop import plan as crop_adapter

    job = jobs.at(Path(jobs_dir), job_id)
    progress = job.progress_path

    work = Path(cfg["work"])  # live 세션 dir — detected.json/preview.mp4 위치
    src = work / "preview.mp4"
    out = work / "render.mp4"
    tmp = work / "._render_raw.mp4"
    overrides = cfg.get("overrides") or {}
    toggles = cfg.get("toggles") or {}
    show_boxes = bool(toggles.get("obj_boxes", True))
    show_trails = bool(toggles.get("show_trails", True))
    draw_crop_box = bool(toggles.get("draw_crop_box", True))
    show_dead_zone = bool(toggles.get("show_dead_zone", True))
    show_center_line = bool(toggles.get("show_center_line", True))
    show_highlight = bool(toggles.get("show_highlight", True))

    try:
        video.require_ffmpeg()
        if not src.exists():
            raise RuntimeError("Preview video not found — run detection first.")

        detected = load_detections(
            _json.loads((work / "detected.json").read_text(encoding="utf-8"))
        )
        # 좌표는 검출 당시의 원본 해상도 기준으로 계산한다 (meta.json). validate=False:
        # 튜닝 노브 조합 하나가 렌더 잡을 죽이지 않게 — 여기 결과는 미리보기다.
        meta = _json.loads((work / "meta.json").read_text(encoding="utf-8"))
        vinfo = crop_adapter.video_info_from_meta(meta)
        cfg_resolved = crop_adapter.resolve_clip_config(overrides)
        cropres = plan_from_detections(
            detected,
            crop_adapter.crop_spec_for(vinfo.width, vinfo.height),
            vinfo,
            config=cfg_resolved,
            collect_debug=show_highlight,
            validate=False,
        )

        vmeta = video.probe(src)
        fps, total, w, h = vmeta.fps, vmeta.frame_count, vmeta.width, vmeta.height
        cap = cv2.VideoCapture(str(src))

        # 좌표는 원본 해상도 기준 — 프리뷰가 다른 크기로 트랜스코딩됐으면 맞춰 스케일
        scale = w / float(vinfo.width or w)

        traj = crop.geometry.build_trajectory(cropres.samples, vinfo.width)
        traj = (traj[0], [x * scale for x in traj[1]])
        types = crop.geometry.build_types(cropres.samples)
        crop_w = crop.geometry.crop_width_for(h, w)

        # 디버그 bbox 도 프리뷰 해상도로 맞춰 둔다 — 프레임마다 다시 계산하지 않게
        # 루프 밖에서 한 번만 (ms 에 의존하지 않는 변환이다).
        debug_lookup = None
        if cropres.debug:
            entries = cropres.debug
            if abs(scale - 1.0) > 1e-6:
                entries = [
                    {
                        "video_offset_ms": e["video_offset_ms"],
                        "ball_bbox": [v * scale for v in e["ball_bbox"]] if e.get("ball_bbox") else None,
                        "carrier_bbox": [v * scale for v in e["carrier_bbox"]] if e.get("carrier_bbox") else None,
                    }
                    for e in cropres.debug
                ]
            debug_lookup = crop.geometry.build_debug_lookup(entries)
        dead_zone_half = cfg_resolved.dead_zone_half * scale

        det_ms = [ms for ms, _ in detected]
        trail_ms = 2000  # 궤적 길이 — 최근 2초

        def _draw_trails(frame, now_ms: float) -> None:
            """선수(track_id별)·선정 공(디버그 궤적)의 최근 2초 이동 경로."""
            lo = bisect.bisect_left(det_ms, now_ms - trail_ms)
            hi = bisect.bisect_right(det_ms, now_ms)
            paths: dict[int, list[tuple[int, int]]] = {}
            for j in range(lo, hi):
                for d in detected[j][1]:
                    if d.track_id is None or d.object_type != "player":
                        continue
                    paths.setdefault(d.track_id, []).append(
                        (
                            int((d.bbox_x + d.bbox_width / 2) * scale),
                            int((d.bbox_y + d.bbox_height / 2) * scale),
                        )
                    )
            for tid, pts in paths.items():
                color = _BOX_PALETTE[tid % len(_BOX_PALETTE)]
                for k in range(1, len(pts)):
                    fade = 0.25 + 0.75 * k / len(pts)
                    cv2.line(
                        frame, pts[k - 1], pts[k],
                        tuple(int(c * fade) for c in color), 2, cv2.LINE_AA,
                    )
            if cropres.debug:
                ball_pts = [
                    (
                        int((e["ball_bbox"][0] + e["ball_bbox"][2] / 2) * scale),
                        int((e["ball_bbox"][1] + e["ball_bbox"][3] / 2) * scale),
                    )
                    for e in cropres.debug
                    if e.get("ball_bbox") and now_ms - trail_ms <= e["video_offset_ms"] <= now_ms
                ]
                for k in range(1, len(ball_pts)):
                    fade = 0.25 + 0.75 * k / len(ball_pts)
                    cv2.line(
                        frame, ball_pts[k - 1], ball_pts[k],
                        tuple(int(c * fade) for c in (92, 92, 255)), 3, cv2.LINE_AA,
                    )

        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        jobs.emit(progress, {"phase": "start", "total": total})

        idx = 0
        cancelled = False
        try:
            while True:
                if job.cancelled():
                    cancelled = True
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                ms = idx / fps * 1000.0

                if show_trails and det_ms:
                    _draw_trails(frame, ms)

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

                cx = crop.geometry.center_at(ms, traj)  # 보간은 프레임당 한 번
                if draw_crop_box and cx is not None:
                    crop.window.draw(frame, cx, crop_w, w, h)
                    crop.hud.draw(
                        frame, cx, w, h,
                        target_type=crop.geometry.type_at(ms, types),
                        dead_zone_half=dead_zone_half,
                        show_dead_zone=show_dead_zone,
                        show_center_line=show_center_line,
                    )
                if debug_lookup is not None:
                    crop.highlight.draw(
                        frame, crop.geometry.debug_at(ms, debug_lookup), w, h
                    )

                writer.write(frame)
                idx += 1
                if idx % 10 == 0 or idx == total:
                    jobs.emit(progress, {"phase": "render", "done": idx, "total": total})
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
    except Exception as e:
        jobs.emit(progress, {"phase": "error", "msg": str(e)})
        raise

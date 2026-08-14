"""연구실 크롭 워커 — 런 하나로 **셋을 한 번에** 낸다.

자식 프로세스에서 돈다(torch·ultralytics·cv2). 진행률은 `jobs_dir/{job_id}/progress.jsonl`.

    검출 1회  →  좌표 계산  →  crop.json  →  한 루프로 wide.mp4 · crop.mp4

셋은 전부 필수다. 선택이 아니다.

  crop.json   좌표. `validate=True` — 파일로 나가는 결과라 규칙 위반이면 실패시킨다.
  wide.mp4    가로. 그리기 옵션이 하나라도 켜져 있으면 오버레이 렌더본, 하나도 없으면
              원본을 하드링크(실패 시 복사)한 사본.
  crop.mp4    세로 크롭. **오버레이를 굽지 않는다** — 이건 산출물이지 관찰용이 아니다.

한 번만 디코드한다. 프레임을 읽으면 먼저 깨끗한 상태로 세로를 잘라 내고(crop.mp4),
그 다음에 같은 프레임 위에 오버레이를 그려 가로에 쓴다(wide.mp4). 순서가 뒤집히면
산출물에 HUD 가 박힌다.

cv2.VideoWriter 의 H.264 지원은 OpenCV 빌드마다 달라 믿을 수 없다. 그래서 항상 mp4v
중간 파일을 쓰고 ffmpeg 로 다시 인코딩한다.

**원본은 지우지 않는다.** 연구실 아카이브의 영상이고, 다른 런이 또 쓴다.
"""

from __future__ import annotations

import bisect
from pathlib import Path

from infra import jobs
from lib import crop, video

# BGR — 검출 박스 (선수: track_id 팔레트, 공: 주황)
_BOX_PALETTE = [
    (247, 171, 77), (102, 207, 81), (107, 107, 255), (59, 212, 255), (232, 93, 204),
    (151, 201, 32), (43, 146, 255), (172, 131, 247), (252, 143, 116), (75, 227, 169),
]
_BALL_COLOR = (0, 140, 255)
_TRAIL_MS = 2000  # 궤적 길이 — 최근 2초


class _Cancelled(Exception):
    """검출 콜백에서 CANCEL 센티널을 봤을 때."""


def _detect(cfg: dict, src: Path, device: str, interval_ms: int, job, progress, total: int):
    """검출 패스 — 이 런에서 가장 비싼 부분. 100ms 격자 샘플을 낸다."""
    from adaptive_crop import build_detector, detect_video

    from lib.crop import plan as crop_adapter

    entries = cfg.get("detectors") or []
    if not entries:
        raise RuntimeError("No detection model selected.")
    conf_cfg = cfg.get("conf")
    default_conf = float(conf_cfg) if conf_cfg is not None else crop_adapter.DEFAULT_CROP_CONF
    detector = build_detector(crop_adapter.detector_entries(entries, default_conf), device)

    def on_progress(done: int) -> None:
        if job.cancelled():
            raise _Cancelled()
        if done % 10 == 0 or done >= total:
            jobs.emit(progress, {"phase": "detect", "done": done, "total": total})

    return detect_video(
        str(src), detector=detector, sampling_interval_ms=interval_ms, on_progress=on_progress
    )


def _trail_drawer(detected, det_ms, debug_entries, *, players: bool, ball: bool):
    """선수(track_id별)·선정 공의 최근 2초 이동 경로를 그리는 함수를 만든다."""
    import cv2

    def draw(frame, now_ms: float) -> None:
        if players:
            lo = bisect.bisect_left(det_ms, now_ms - _TRAIL_MS)
            hi = bisect.bisect_right(det_ms, now_ms)
            paths: dict[int, list[tuple[int, int]]] = {}
            for j in range(lo, hi):
                for d in detected[j][1]:
                    if d.track_id is None or d.object_type != "player":
                        continue
                    paths.setdefault(d.track_id, []).append(
                        (int(d.bbox_x + d.bbox_width / 2), int(d.bbox_y + d.bbox_height / 2))
                    )
            for tid, pts in paths.items():
                color = _BOX_PALETTE[tid % len(_BOX_PALETTE)]
                for k in range(1, len(pts)):
                    fade = 0.25 + 0.75 * k / len(pts)
                    cv2.line(
                        frame, pts[k - 1], pts[k],
                        tuple(int(c * fade) for c in color), 2, cv2.LINE_AA,
                    )
        if ball and debug_entries:
            pts = [
                (
                    int(e["ball_bbox"][0] + e["ball_bbox"][2] / 2),
                    int(e["ball_bbox"][1] + e["ball_bbox"][3] / 2),
                )
                for e in debug_entries
                if e.get("ball_bbox") and now_ms - _TRAIL_MS <= e["video_offset_ms"] <= now_ms
            ]
            for k in range(1, len(pts)):
                fade = 0.25 + 0.75 * k / len(pts)
                cv2.line(
                    frame, pts[k - 1], pts[k],
                    tuple(int(c * fade) for c in (92, 92, 255)), 3, cv2.LINE_AA,
                )

    return draw


def _box_drawer(detected, det_ms):
    """직전 샘플의 검출 박스 + 라벨."""
    import cv2

    def draw(frame, now_ms: float) -> None:
        i = min(max(bisect.bisect_right(det_ms, now_ms) - 1, 0), len(detected) - 1)
        for d in detected[i][1]:
            x1, y1 = int(d.bbox_x), int(d.bbox_y)
            x2, y2 = int(d.bbox_x + d.bbox_width), int(d.bbox_y + d.bbox_height)
            if d.object_type == "ball":
                color, label = _BALL_COLOR, f"ball {d.confidence:.0%}"
            else:
                color = _BOX_PALETTE[(d.track_id or 0) % len(_BOX_PALETTE)]
                label = f"#{d.track_id} {d.confidence:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

    return draw


def run_lab_crop(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    import cv2

    from app.core.config import resolve_device
    from app.services.lab_crop_runs import CROP_NAME, CUT_NAME, WIDE_NAME, link_or_copy

    job = jobs.at(Path(jobs_dir), job_id)
    progress = job.progress_path

    src = Path(cfg["source"])  # 연구실 아카이브의 원본 — 지우지 않는다
    work = Path(cfg["work"])  # labs/{lab}/crops/{crop}/
    work.mkdir(parents=True, exist_ok=True)

    toggles = cfg.get("toggles") or {}
    show_boxes = bool(toggles.get("obj_boxes"))
    show_player_trails = bool(toggles.get("player_trails"))
    show_ball_trail = bool(toggles.get("ball_trail"))
    show_highlight = bool(toggles.get("target_highlight"))
    draw_crop_box = bool(toggles.get("crop_box"))
    show_dead_zone = bool(toggles.get("dead_zone"))
    show_center_line = bool(toggles.get("center_line"))
    overlay = any(
        (show_boxes, show_player_trails, show_ball_trail, show_highlight,
         draw_crop_box, show_dead_zone, show_center_line)
    )

    # 크롭 창은 절대 px 다 — 소스보다 크면 소스 크기로 clamp 된다
    req_w = int(cfg.get("crop_w") or crop.geometry.DEFAULT_CROP_W)
    req_h = int(cfg.get("crop_h") or crop.geometry.DEFAULT_CROP_H)

    try:
        video.require_ffmpeg()  # 한 시간 렌더한 뒤 마지막에 실패하지 않도록 먼저

        # adaptive_crop 은 cv2/ultralytics 를 끌고 온다 — 워커 안에서만 import
        from adaptive_crop import VideoInfo, plan_from_detections

        from lib.crop import plan as crop_adapter

        meta = video.probe(src)
        fps, total, w, h = meta.fps, meta.frame_count, meta.width, meta.height
        clip_cfg = crop_adapter.resolve_clip_config(cfg.get("overrides") or None)
        interval = clip_cfg.sampling_interval_ms
        sample_total = max(1, (meta.duration_ms // interval) + 1)

        jobs.emit(progress, {"phase": "start", "total": total})

        # ---- 검출 (비싼 패스) ----
        try:
            detected = _detect(
                cfg, src, resolve_device(cfg.get("device")), interval,
                job, progress, sample_total,
            )
        except _Cancelled:
            jobs.emit(progress, {"phase": "cancelled", "total": total})
            return {"status": "cancelled"}

        # ---- 좌표 ----
        jobs.emit(progress, {"phase": "crop_analyze", "total": total})
        # 컨테이너가 프레임 수를 안 들고 있으면(0) 샘플 격자의 끝을 길이로 본다
        duration_ms = meta.duration_ms or (detected[-1][0] if detected else 0)
        vinfo = VideoInfo(width=w, height=h, fps=fps, duration_ms=duration_ms)
        result = plan_from_detections(
            detected,
            crop_adapter.crop_spec_for(w, h, req_w, req_h),
            vinfo,
            config=clip_cfg,
            collect_debug=show_highlight or show_ball_trail,
        )
        (work / CROP_NAME).write_text(crop_adapter.crop_plan_json(result), encoding="utf-8")

        traj = crop.geometry.build_trajectory(result.samples, w)
        targets = crop.geometry.build_targets(result.samples)
        debug_lookup = (
            crop.geometry.build_debug_lookup(result.debug) if result.debug else None
        )
        crop_w, crop_h, crop_y = crop.geometry.crop_window_for(w, h, req_w, req_h)

        det_ms = [ms for ms, _ in detected]
        draw_trails = (
            _trail_drawer(
                detected, det_ms, result.debug,
                players=show_player_trails, ball=show_ball_trail,
            )
            if (show_player_trails or show_ball_trail) and det_ms
            else None
        )
        draw_boxes = _box_drawer(detected, det_ms) if show_boxes and det_ms else None

        # ---- 한 루프로 둘을 쓴다 ----
        wide_tmp = work / "._wide_raw.mp4"
        cut_tmp = work / "._cut_raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cut_writer = cv2.VideoWriter(str(cut_tmp), fourcc, fps, (crop_w, crop_h))
        wide_writer = cv2.VideoWriter(str(wide_tmp), fourcc, fps, (w, h)) if overlay else None

        cap = cv2.VideoCapture(str(src))
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
                cx = crop.geometry.center_at(ms, traj)  # 보간은 프레임당 한 번

                # 세로부터 — 깨끗한 프레임이어야 한다
                cut_writer.write(crop.cut.window(frame, cx, crop_w, w, crop_h, crop_y))

                if wide_writer is not None:
                    if draw_trails is not None:
                        draw_trails(frame, ms)
                    if draw_boxes is not None:
                        draw_boxes(frame, ms)
                    if cx is not None and (draw_crop_box or show_dead_zone or show_center_line):
                        if draw_crop_box:
                            crop.window.draw(frame, cx, crop_w, w, h, crop_h, crop_y)
                        target = crop.geometry.target_at(ms, targets)
                        crop.hud.draw(
                            frame, cx, w, h,
                            target_type=target[0] if target else None,
                            confidence=target[1] if target else None,
                            dead_zone_half=clip_cfg.dead_zone_half,
                            show_dead_zone=show_dead_zone,
                            show_center_line=show_center_line,
                        )
                    if show_highlight and debug_lookup is not None:
                        crop.highlight.draw(
                            frame, crop.geometry.debug_at(ms, debug_lookup), w, h
                        )
                    wide_writer.write(frame)

                idx += 1
                if idx % 10 == 0 or idx == total:
                    jobs.emit(progress, {"phase": "render", "done": idx, "total": total})
        finally:
            cap.release()
            cut_writer.release()
            if wide_writer is not None:
                wide_writer.release()

        if cancelled:
            cut_tmp.unlink(missing_ok=True)
            wide_tmp.unlink(missing_ok=True)
            jobs.emit(progress, {"phase": "cancelled", "done": idx, "total": total})
            return {"status": "cancelled"}

        # ---- 인코딩 ----
        jobs.emit(progress, {"phase": "encoding", "done": idx, "total": total})
        video.to_h264(cut_tmp, work / CUT_NAME)
        cut_tmp.unlink(missing_ok=True)
        if overlay:
            video.to_h264(wide_tmp, work / WIDE_NAME)
            wide_tmp.unlink(missing_ok=True)
            wide_kind = "render"
        else:
            # 그릴 것이 없으면 원본 그대로가 곧 가로 영상이다 — 하드링크면 디스크는
            # 한 벌만 쓰고, 아카이브에서 원본을 지워도 링크는 살아 있다.
            wide_kind = link_or_copy(src, work / WIDE_NAME)

        jobs.emit(
            progress,
            {"phase": "done", "done": idx, "total": total, "wide_kind": wide_kind},
        )
        return {"status": "done", "frames": idx, "wide_kind": wide_kind}
    except Exception as e:
        jobs.emit(progress, {"phase": "error", "msg": str(e)})
        raise

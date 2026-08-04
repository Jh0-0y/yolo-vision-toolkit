"""클립 플래너 — 트랙 확정(balltrack) + 타깃 결정 + 경로 최적화(pathopt).

클립 전체를 한 번에 보고 크롭 경로를 계산하는 2-패스 경로:

  detect  →  select_game_ball_track  →  타깃 결정(보간·fallback)  →  optimize_path

순차(프레임 단위) 방식과의 핵심 차이는 공 미검출 구간이 "외삽(추측)"이 아니라
"보간(확정)"이라는 것 — 재등장 위치를 이미 알기 때문에 크롭 점프가 구조적으로 없다.
"""

import bisect

from .balltrack import (
    BallTrack,
    ClipPlanConfig,
    ball_in_player_box,
    player_group_center,
    select_game_ball_track,
)
from .constants import CENTER_FALLBACK_X, CROP_WIDTH, PLAYER_LOST_HOLD_MS
from .pathopt import optimize_path
from .types import Detection, TargetSample, TargetType


def _bbox(det: Detection | None) -> list[float] | None:
    if det is None:
        return None
    return [det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height]


def _hermite(
    p1: float, p2: float, m1: float, m2: float, s: float
) -> float:
    """큐빅 Hermite 보간 — 값 p1→p2, 구간 스케일 접선 m1·m2, s∈[0,1]."""
    s2 = s * s
    s3 = s2 * s
    return (
        (2 * s3 - 3 * s2 + 1) * p1
        + (s3 - 2 * s2 + s) * m1
        + (-2 * s3 + 3 * s2) * p2
        + (s3 - s2) * m2
    )


def _tangent(
    prev_v: float | None, prev_t: int | None,
    v1: float, t1: int, v2: float, t2: int,
    next_v: float | None, next_t: int | None,
    at_start: bool,
) -> float:
    """비균등 시간 Catmull-Rom 접선 (구간 [t1,t2] 스케일). 이웃이 없으면 현(chord).

    과도한 오버슛 방지를 위해 현 길이의 2배로 클램프한다.
    """
    span = t2 - t1
    chord = v2 - v1
    if at_start:
        m = (v2 - prev_v) / (t2 - prev_t) * span if prev_v is not None else chord
    else:
        m = (next_v - v1) / (next_t - t1) * span if next_v is not None else chord
    limit = 2 * abs(chord) + 1e-9
    return max(-limit, min(limit, m))


def _ball_at(
    track: BallTrack | None, offset_ms: int, cfg: ClipPlanConfig
) -> tuple[float, float, list[float]] | None:
    """시각 offset_ms의 공 위치. (x, confidence, bbox [x,y,w,h]) 또는 None.

    실측 사이 gap ≤ interp_max_gap_ms 이면 보간 — 양끝을 아는 확정 경로다.
    직선(선형)이 아니라 이웃 점의 진행 방향을 반영한 곡선(Catmull-Rom/Hermite)으로
    메꾼다: 미검출 구간이 실제 궤적(포물선·턴)을 직선으로 잘라먹지 않게.
    bbox도 함께 보간해 하이라이트 마커·궤적이 실측 사이에서 끊기지 않는다.
    트랙 범위 밖이거나 긴 gap 안이면 None (fallback으로 넘어감).
    """
    if track is None or not track.points:
        return None
    times = [p.offset_ms for p in track.points]
    i = bisect.bisect_left(times, offset_ms)
    if i < len(times) and times[i] == offset_ms:
        det = track.points[i].det
        return det.center_x, det.confidence, _bbox(det)
    if i == 0 or i == len(times):
        return None  # 트랙 시작 전 / 종료 후
    before, after = track.points[i - 1], track.points[i]
    gap = after.offset_ms - before.offset_ms
    if gap > cfg.interp_max_gap_ms:
        return None  # 장기 가림 — 보간 대신 소유선수 fallback
    s = (offset_ms - before.offset_ms) / gap

    prev_pt = track.points[i - 2] if i >= 2 else None
    next_pt = track.points[i + 1] if i + 1 < len(times) else None
    bd, ad = before.det, after.det
    t1, t2 = before.offset_ms, after.offset_ms

    def curve(get) -> float:
        v1, v2 = get(bd), get(ad)
        pv = get(prev_pt.det) if prev_pt else None
        nv = get(next_pt.det) if next_pt else None
        m1 = _tangent(pv, prev_pt.offset_ms if prev_pt else None, v1, t1, v2, t2, None, None, True)
        m2 = _tangent(None, None, v1, t1, v2, t2, nv, next_pt.offset_ms if next_pt else None, False)
        return _hermite(v1, v2, m1, m2, s)

    cx = curve(lambda d: d.center_x)
    cy = curve(lambda d: d.center_y)
    # 크기는 위치만큼 민감하지 않다 — 선형
    w = bd.bbox_width + (ad.bbox_width - bd.bbox_width) * s
    h = bd.bbox_height + (ad.bbox_height - bd.bbox_height) * s
    bbox = [cx - w / 2, cy - h / 2, w, h]
    conf = min(bd.confidence, ad.confidence)
    return cx, conf, bbox


def _build_carrier_timeline(
    track: BallTrack | None,
    players_at: dict[int, list[Detection]],
    cfg: ClipPlanConfig,
) -> list[tuple[int, int]]:
    """실측 공 점마다 기하 겹침으로 소유선수를 판정한 (ms, track_id) 시퀀스."""
    if track is None:
        return []
    timeline: list[tuple[int, int]] = []
    for point in track.points:
        overlapping = [
            p
            for p in players_at.get(point.offset_ms, [])
            if p.track_id is not None
            and ball_in_player_box(point.det, p, cfg.possession_margin)
        ]
        if overlapping:
            nearest = min(
                overlapping, key=lambda p: abs(p.center_x - point.det.center_x)
            )
            timeline.append((point.offset_ms, nearest.track_id))
    return timeline


def _carrier_as_of(timeline: list[tuple[int, int]], offset_ms: int) -> int | None:
    """offset_ms 이전 마지막 소유선수 track_id (없으면 None)."""
    i = bisect.bisect_right(timeline, (offset_ms, float("inf")))
    return timeline[i - 1][1] if i > 0 else None


def resolve_targets_clip(
    samples: list[tuple[int, list[Detection]]],
    cfg: ClipPlanConfig | None = None,
    debug: list | None = None,
) -> tuple[list[TargetSample], dict]:
    """클립 전체를 놓고 시점별 Target X를 결정한다 (최적화 전 원시 타깃).

    우선순위: 공(실측/보간) → 공+군집 가중 중심 → 소유선수 → 군집 → 직전 유지 → 중앙.
    """
    cfg = cfg or ClipPlanConfig()
    track, info = select_game_ball_track(samples, cfg)
    players_at = {
        offset_ms: [d for d in dets if d.object_type == "player"]
        for offset_ms, dets in samples
    }
    carrier_timeline = _build_carrier_timeline(track, players_at, cfg)

    results: list[TargetSample] = []
    last_valid_x = float(CENTER_FALLBACK_X)
    last_confidence = 0.0
    last_observed_ms: int | None = None

    for offset_ms, detections in samples:
        players = players_at[offset_ms]
        group_x = player_group_center(detections)
        ball = _ball_at(track, offset_ms, cfg)

        if ball is not None or group_x is not None:
            last_observed_ms = offset_ms

        carrier_det: Detection | None = None
        if ball is not None:
            ball_x, confidence, ball_bbox = ball
            if group_x is not None and abs(ball_x - group_x) <= CROP_WIDTH:
                target_x = ball_x * cfg.ball_weight + group_x * cfg.player_group_weight
                target_type = TargetType.BALL_PLAYER
            else:
                target_x = ball_x
                target_type = TargetType.BALL
        else:
            ball_bbox = None
            # use_carrier=False면 소유선수 추적 생략 — 크롭이 특정 선수로 튀지 않게
            carrier_id = _carrier_as_of(carrier_timeline, offset_ms) if cfg.use_carrier else None
            carrier_det = (
                next((p for p in players if p.track_id == carrier_id), None)
                if carrier_id is not None
                else None
            )
            if carrier_det is not None:
                # 장기 가림 — 공을 마지막으로 든 선수를 따라간다
                target_x = carrier_det.center_x
                target_type = TargetType.PLAYER_GROUP
                confidence = carrier_det.confidence
            elif group_x is not None:
                target_x = group_x
                target_type = TargetType.PLAYER_GROUP
                confidence = max(p.confidence for p in players)
            elif (
                last_observed_ms is not None
                and offset_ms - last_observed_ms <= PLAYER_LOST_HOLD_MS
            ):
                target_x = last_valid_x
                target_type = TargetType.PLAYER_GROUP
                confidence = last_confidence
            else:
                target_x = float(CENTER_FALLBACK_X)
                target_type = TargetType.CENTER
                confidence = 0.0

        last_valid_x = target_x
        last_confidence = confidence
        results.append(
            TargetSample(
                video_offset_ms=offset_ms,
                target_center_x=target_x,
                target_type=target_type.value,
                confidence=confidence,
            )
        )
        if debug is not None:
            debug.append(
                {
                    "video_offset_ms": offset_ms,
                    "ball_bbox": ball_bbox,  # 실측+보간 — 마커가 끊기지 않는다
                    "carrier_bbox": _bbox(carrier_det),
                }
            )

    return results, info


def plan_clip(
    samples: list[tuple[int, list[Detection]]],
    cfg: ClipPlanConfig | None = None,
    debug: list | None = None,
) -> tuple[list[TargetSample], dict]:
    """검출 시퀀스 → 최종 크롭 중심 경로 (타깃 결정 + 전역 최적화)."""
    cfg = cfg or ClipPlanConfig()
    targets, info = resolve_targets_clip(samples, cfg, debug)
    path = optimize_path(targets, cfg)
    planned = [
        TargetSample(
            video_offset_ms=t.video_offset_ms,
            target_center_x=x,
            target_type=t.target_type,
            confidence=t.confidence,
        )
        for t, x in zip(targets, path)
    ]
    return planned, info

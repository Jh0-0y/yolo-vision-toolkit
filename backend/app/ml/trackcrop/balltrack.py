"""클립 단위 공 트랙 확정 (2-패스 설계의 패스 1).

프레임 단위 공 선정 대신, 클립 전체를 놓고 트랙(궤적) 단위로 판단한다:

  1. tracklet 구성 — ByteTrack id로 묶되, 같은 id 안의 불가능한 점프는 쪼갠다
  2. 스티칭      — 조각 끝/시작의 위치·속도·시간 일관성으로 클립 전체에서 연결
  3. 트랙 채점    — 이동량·선수 상호작용(기하 겹침)·커버 시간으로 경기 공 채택
  4. 흡수 병합    — 시간이 안 겹치는 나머지 트랙을 완화된 허용치로 회수

관중석·벤치·프린트 공은 궤적이 죽어 있고(이동량↓) 선수와 상호작용이 없어
트랙 단위에서 명확히 갈린다. 설계 배경은 docs/trackcrop-clip-planner.md 참고.
"""

import statistics
from dataclasses import dataclass, field, fields

from . import constants
from .types import Detection

# ---------------------------------------------------------------------------
# 설정


@dataclass(frozen=True)
class ClipPlanConfig:
    """클립 플래너 튜닝 값 — 파이프라인의 유일한 런타임 설정."""

    # --- 검출/지오메트리 정책 ---
    sampling_interval_ms: int = constants.SAMPLING_INTERVAL_MS  # 100 (detect 단계)
    dead_zone_width: float = constants.DEAD_ZONE_WIDTH  # 208 — 무페널티 구간 폭
    max_move_px_per_second: float = constants.MAX_MOVE_PX_PER_SECOND  # 1200 — 크롭 속도 캡
    ball_weight: float = constants.BALL_WEIGHT  # 0.7 — 공+군집 가중 (군집 = 1-이 값)

    # --- tracklet 쪼개기 / 스티칭 ---
    ball_max_speed_px_s: float = 3000.0  # 공 물리 한계 속도 (슛 포함 — 크롭 속도와 무관)
    split_base_px: float = 80.0  # 같은 id 내 허용 점프 base
    stitch_max_gap_ms: int = 2500  # 조각 연결 최대 시간 간격
    stitch_base_px: float = 150.0  # 조각 연결 허용 거리 base
    stitch_velocity_cap_ms: int = 500  # 스티칭 외삽에 쓰는 속도 적용 상한(과외삽 방지)

    # --- 트랙 채점 ---
    # 이동량이 1등 신호다: 진짜 경기 공은 클립 전체에서 압도적으로 많이 움직인다.
    # 상호작용 비중을 크게 두면 "드리블 구간만 모은 부분 체인"이나 머리 오탐
    # 체인(항상 선수와 겹침)이 전체 궤적을 이기는 역전이 생긴다 (실측 사례).
    w_travel: float = 0.5  # 총 이동량
    w_interaction: float = 0.3  # 선수 상호작용 비율 (기하 겹침)
    w_span: float = 0.1  # 클립 대비 커버 시간
    travel_norm_px: float = 3840.0  # 포화 완화 — 1920이면 부분 체인도 만점이라 변별력 상실
    min_track_score: float = 0.25  # 미달이면 "공 없음" 클립
    absorb_allow_scale: float = 2.0  # 흡수 병합 시 허용치 완화 배율
    possession_margin: float = 0.15  # 소유 판정: 선수 bbox 확장 비율 (×키)

    # --- 타깃 결정 ---
    interp_max_gap_ms: int = 2000  # 이하 gap은 선형 보간(BALL 유지)
    use_carrier: bool = False  # 공 없을 때 소유선수(특정 선수) 추적 여부. off=선수군집/유지로

    # --- 경로 최적화 ---
    w_follow: float = 1.0  # 데드존 밖 이탈 페널티
    w_inside: float = 0.05  # 데드존 안 미세 인력 (수치 안정 + 중심 회귀)
    w_vel: float = 0.1  # 팬 속도 페널티
    w_acc: float = 15.0  # 팬 가속 페널티
    irls_iters: int = 30  # IRLS 반복 횟수
    min_follow_conf: float = 0.1  # confidence 가중 하한

    @property
    def dead_zone_half(self) -> float:
        return self.dead_zone_width / 2

    @property
    def player_group_weight(self) -> float:
        return 1.0 - self.ball_weight


_FIELD_NAMES = {f.name for f in fields(ClipPlanConfig)}


def resolve_clip_config(overrides: dict | None) -> ClipPlanConfig:
    """overrides(dict) 중 알려진 필드·비어있지 않은 값만 반영해 Config를 만든다.

    알 수 없는 키·None 값은 무시한다 (프론트가 빈 노브를 안 보내거나 None으로 보냄).
    """
    if not overrides:
        return ClipPlanConfig()
    clean = {k: v for k, v in overrides.items() if k in _FIELD_NAMES and v is not None}
    return ClipPlanConfig(**clean)


def player_group_center(detections: list[Detection]) -> float | None:
    """선수 군집 중심 X — 중앙값 (심판·벤치 등 outlier에 강건). 선수 없으면 None."""
    xs = [d.center_x for d in detections if d.object_type == "player"]
    if not xs:
        return None
    return float(statistics.median(xs))


# ---------------------------------------------------------------------------
# 트랙 자료구조


@dataclass
class TrackPoint:
    offset_ms: int
    det: Detection


@dataclass
class BallTrack:
    """시간순 공 관측 시퀀스 — tracklet이자 스티칭 결과."""

    points: list[TrackPoint] = field(default_factory=list)

    @property
    def start_ms(self) -> int:
        return self.points[0].offset_ms

    @property
    def end_ms(self) -> int:
        return self.points[-1].offset_ms

    @property
    def start_x(self) -> float:
        return self.points[0].det.center_x

    @property
    def end_x(self) -> float:
        return self.points[-1].det.center_x

    def tail_velocity(self, window: int = 3) -> float:
        """끝쪽 window개 점의 평균 속도 (px/ms) — 스티칭 외삽용."""
        pts = self.points[-window:]
        if len(pts) < 2:
            return 0.0
        dt = pts[-1].offset_ms - pts[0].offset_ms
        if dt <= 0:
            return 0.0
        return (pts[-1].det.center_x - pts[0].det.center_x) / dt

    def travel_px(self) -> float:
        """총 경로 길이(x) — 정지 공 판별의 핵심 신호."""
        return sum(
            abs(b.det.center_x - a.det.center_x)
            for a, b in zip(self.points, self.points[1:])
        )

    def overlaps_time(self, other: "BallTrack") -> bool:
        return not (self.end_ms < other.start_ms or other.end_ms < self.start_ms)


# ---------------------------------------------------------------------------
# 1. tracklet 구성


def build_tracklets(
    samples: list[tuple[int, list[Detection]]], cfg: ClipPlanConfig
) -> list[BallTrack]:
    """공 검출을 track_id로 묶고, 같은 id 안의 불가능한 점프에서 쪼갠다.

    id 없는 검출은 단독 tracklet — 스티칭 단계에서 이어붙일 수 있다.
    """
    by_id: dict[int, list[TrackPoint]] = {}
    singles: list[BallTrack] = []
    for offset_ms, detections in samples:
        for det in detections:
            if det.object_type != "ball":
                continue
            point = TrackPoint(offset_ms, det)
            if det.track_id is None:
                singles.append(BallTrack([point]))
            else:
                by_id.setdefault(det.track_id, []).append(point)

    tracklets: list[BallTrack] = []
    for points in by_id.values():
        points.sort(key=lambda p: p.offset_ms)
        current = [points[0]]
        for prev, point in zip(points, points[1:]):
            dt = point.offset_ms - prev.offset_ms
            allow = cfg.split_base_px + cfg.ball_max_speed_px_s * dt / 1000
            if abs(point.det.center_x - prev.det.center_x) > allow:
                tracklets.append(BallTrack(current))  # id가 오탐을 문 지점 — 쪼갬
                current = [point]
            else:
                current.append(point)
        tracklets.append(BallTrack(current))
    tracklets.extend(singles)
    tracklets.sort(key=lambda t: t.start_ms)
    return tracklets


# ---------------------------------------------------------------------------
# 2. 스티칭


def _link_allowance(gap_ms: int, cfg: ClipPlanConfig, scale: float = 1.0) -> float:
    return scale * (cfg.stitch_base_px + cfg.ball_max_speed_px_s * gap_ms / 1000)


def _link_distance(chain: BallTrack, nxt: BallTrack, gap_ms: int, cfg: ClipPlanConfig) -> float:
    """앞 트랙 끝에서 외삽한 예측과 뒤 트랙 시작의 거리."""
    extrapolate_ms = min(gap_ms, cfg.stitch_velocity_cap_ms)
    predicted = chain.end_x + chain.tail_velocity() * extrapolate_ms
    return abs(nxt.start_x - predicted)


def stitch_tracks(tracklets: list[BallTrack], cfg: ClipPlanConfig) -> list[BallTrack]:
    """시간순으로 tracklet을 체인에 연결한다.

    시간이 겹치는 조각은 절대 같은 체인이 되지 않는다 (동시에 보이면 다른 공).
    """
    chains: list[BallTrack] = []
    for tracklet in tracklets:
        best: BallTrack | None = None
        best_dist = float("inf")
        for chain in chains:
            gap = tracklet.start_ms - chain.end_ms
            if gap <= 0 or gap > cfg.stitch_max_gap_ms:
                continue
            dist = _link_distance(chain, tracklet, gap, cfg)
            if dist <= _link_allowance(gap, cfg) and dist < best_dist:
                best, best_dist = chain, dist
        if best is not None:
            best.points.extend(tracklet.points)
        else:
            chains.append(BallTrack(list(tracklet.points)))
    return chains


# ---------------------------------------------------------------------------
# 3. 채점 + 4. 흡수 병합


def ball_in_player_box(ball: Detection, player: Detection, margin_ratio: float) -> bool:
    """공 중심이 선수 bbox(+margin) 안에 있는가 — 소유/상호작용 판정 (custom 방식)."""
    margin = player.bbox_height * margin_ratio
    bx = ball.center_x
    by = ball.bbox_y + ball.bbox_height / 2
    return (
        player.bbox_x - margin <= bx <= player.bbox_x + player.bbox_width + margin
        and player.bbox_y - margin <= by <= player.bbox_y + player.bbox_height + margin
    )


def _interaction_fraction(
    track: BallTrack,
    players_at: dict[int, list[Detection]],
    cfg: ClipPlanConfig,
) -> float:
    """트랙 점 중 선수와 겹친(들고 있거나 스친) 비율 — 경기 공의 핵심 신호."""
    if not track.points:
        return 0.0
    hits = sum(
        1
        for p in track.points
        if any(
            ball_in_player_box(p.det, player, cfg.possession_margin)
            for player in players_at.get(p.offset_ms, [])
        )
    )
    return hits / len(track.points)


def score_track(
    track: BallTrack,
    players_at: dict[int, list[Detection]],
    clip_span_ms: int,
    cfg: ClipPlanConfig,
) -> float:
    travel = min(track.travel_px() / cfg.travel_norm_px, 1.0)
    interaction = _interaction_fraction(track, players_at, cfg)
    span = min((track.end_ms - track.start_ms) / max(clip_span_ms, 1), 1.0)
    return cfg.w_travel * travel + cfg.w_interaction * interaction + cfg.w_span * span


def select_game_ball_track(
    samples: list[tuple[int, list[Detection]]],
    cfg: ClipPlanConfig | None = None,
) -> tuple[BallTrack | None, dict]:
    """클립 전체에서 경기 공 트랙을 확정한다. (트랙, 진단 정보) 반환.

    최고점 트랙이 min_track_score 미달이면 (None, info) — "공 없음" 클립.
    """
    cfg = cfg or ClipPlanConfig()
    players_at = {
        offset_ms: [d for d in dets if d.object_type == "player"]
        for offset_ms, dets in samples
    }
    clip_span_ms = samples[-1][0] - samples[0][0] if samples else 0

    chains = stitch_tracks(build_tracklets(samples, cfg), cfg)
    if not chains:
        return None, {"tracks": 0}

    scored = sorted(
        ((score_track(c, players_at, clip_span_ms, cfg), c) for c in chains),
        key=lambda sc: sc[0],
        reverse=True,
    )
    best_score, best = scored[0]
    info = {
        "tracks": len(chains),
        "best_score": round(best_score, 3),
        "scores": [round(s, 3) for s, _ in scored],
    }
    if best_score < cfg.min_track_score:
        return None, info

    # 흡수 병합 — 장기 가림 등으로 스티칭에 실패한 같은 공을 회수한다.
    # 시간이 겹치지 않고, 완화된 허용치로 이어지는 트랙만.
    absorbed = 0
    for _, chain in scored[1:]:
        if chain.overlaps_time(best):
            continue
        earlier, later = (chain, best) if chain.end_ms < best.start_ms else (best, chain)
        gap = later.start_ms - earlier.end_ms
        dist = _link_distance(earlier, later, gap, cfg)
        if dist <= _link_allowance(gap, cfg, scale=cfg.absorb_allow_scale):
            best.points = sorted(
                best.points + chain.points, key=lambda p: p.offset_ms
            )
            absorbed += 1
    info["absorbed"] = absorbed

    # 점 단위 흡수 — 스티칭이 궤적을 두 체인으로 쪼갠 경우(경쟁 leapfrog),
    # 낙선 체인의 점이라도 승자 경로와 시간·위치가 정합하면 회수한다.
    best.points.sort(key=lambda p: p.offset_ms)
    info["absorbed_points"] = _absorb_consistent_points(
        best, [c for _, c in scored[1:]], cfg
    )

    # 같은 시각 중복 점 정리 (병합 안전망) — confidence 높은 쪽 유지
    dedup: dict[int, TrackPoint] = {}
    for p in best.points:
        kept = dedup.get(p.offset_ms)
        if kept is None or p.det.confidence > kept.det.confidence:
            dedup[p.offset_ms] = p
    best.points = [dedup[ms] for ms in sorted(dedup)]
    return best, info


def _absorb_consistent_points(
    best: BallTrack, others: list[BallTrack], cfg: ClipPlanConfig
) -> int:
    """낙선 체인의 점 중 승자 트랙 경로와 정합하는 점을 흡수한다.

    승자 트랙의 이웃 실측에서 예상 위치를 보간(범위 밖이면 끝점 기준)하고,
    시간 간격에 비례한 허용 반경 안이면 같은 공의 관측으로 보고 회수한다.
    """
    import bisect as _bisect

    added = 0
    for chain in others:
        times = [p.offset_ms for p in best.points]
        have = set(times)
        for p in chain.points:
            ms = p.offset_ms
            if ms in have:
                continue
            i = _bisect.bisect_left(times, ms)
            if 0 < i < len(times):
                before, after = best.points[i - 1], best.points[i]
                span = after.offset_ms - before.offset_ms
                ratio = (ms - before.offset_ms) / span
                expected = (
                    before.det.center_x
                    + (after.det.center_x - before.det.center_x) * ratio
                )
                gap = min(ms - before.offset_ms, after.offset_ms - ms)
            elif i == 0:
                expected = best.points[0].det.center_x
                gap = times[0] - ms
            else:
                expected = best.points[-1].det.center_x
                gap = ms - times[-1]
            if abs(p.det.center_x - expected) <= _link_allowance(gap, cfg):
                best.points.insert(i, p)
                times.insert(i, ms)
                have.add(ms)
                added += 1
    return added

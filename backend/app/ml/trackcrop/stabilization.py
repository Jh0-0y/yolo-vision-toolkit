"""좌표 안정화.

파이프라인: Outlier 제거 → 짧은 미검출 구간 선형 보간
          → 최대 이동 속도 제한 → 이동 평균 필터

화면 밖 Bounding Box 보정은 Detection 단계에서 프레임 내 좌표만
생성되므로 여기서는 해당 없다.
"""

import statistics

from .constants import DEAD_ZONE_WIDTH, MAX_MOVE_PX_PER_SECOND, SAMPLING_INTERVAL_MS
from .types import TargetSample

# 구현 세부 상수
_OUTLIER_WINDOW = 5  # 국소 중앙값 창 크기
_OUTLIER_DEVIATION_PX = 300.0  # 중앙값 이탈 허용치
_LOW_CONFIDENCE = 0.3  # 저신뢰 단일 샘플 판정 기준
_SMOOTH_WINDOW = 5  # 이동 평균 창 크기 (중심)

# 샘플 간 최대 이동량: 1200px/s × 0.1s = 120px
_MAX_STEP_PX = MAX_MOVE_PX_PER_SECOND * SAMPLING_INTERVAL_MS / 1000

# 데드존 반폭 — 공이 크롭 중심 ±이 값 안이면 크롭 고정
_DEAD_ZONE_HALF = DEAD_ZONE_WIDTH / 2  # 104


def remove_outliers(samples: list[TargetSample]) -> list[TargetSample]:
    """국소 중앙값에서 과도하게 벗어난 샘플과 저신뢰 이탈 샘플을 제거한다.

    첫·마지막 샘플은 제거하지 않는다 — 첫·마지막 Keyframe 필수 조건 보호.
    """
    if len(samples) <= 2:
        return list(samples)

    xs = [s.target_center_x for s in samples]
    half = _OUTLIER_WINDOW // 2
    kept: list[TargetSample] = []
    for i, sample in enumerate(samples):
        if i == 0 or i == len(samples) - 1:
            kept.append(sample)
            continue
        window = xs[max(0, i - half) : i + half + 1]
        deviation = abs(sample.target_center_x - statistics.median(window))
        if deviation > _OUTLIER_DEVIATION_PX:
            continue
        if sample.confidence < _LOW_CONFIDENCE and deviation > _OUTLIER_DEVIATION_PX / 2:
            continue
        kept.append(sample)
    return kept


def interpolate_gaps(samples: list[TargetSample]) -> list[TargetSample]:
    """Outlier 제거로 빠진 100ms 격자 지점을 선형 보간으로 복원한다.

    x(t) = x0 + (x1 - x0) × (t - t0) / (t1 - t0)
    target_type과 confidence는 시간상 더 가까운 이웃의 값을 쓴다.
    """
    if not samples:
        return []

    result: list[TargetSample] = [samples[0]]
    for prev, nxt in zip(samples, samples[1:], strict=False):
        t0, t1 = prev.video_offset_ms, nxt.video_offset_ms
        t = t0 + SAMPLING_INTERVAL_MS
        while t < t1:
            ratio = (t - t0) / (t1 - t0)
            nearer = prev if (t - t0) <= (t1 - t) else nxt
            result.append(
                TargetSample(
                    video_offset_ms=t,
                    target_center_x=prev.target_center_x
                    + (nxt.target_center_x - prev.target_center_x) * ratio,
                    target_type=nearer.target_type,
                    confidence=nearer.confidence,
                )
            )
            t += SAMPLING_INTERVAL_MS
        result.append(nxt)
    return result


def limit_speed(samples: list[TargetSample]) -> list[TargetSample]:
    """샘플 간 이동을 최대 속도(1200px/s → 120px/100ms) 이내로 제한한다."""
    if not samples:
        return []

    result: list[TargetSample] = [samples[0]]
    for sample in samples[1:]:
        prev_x = result[-1].target_center_x
        delta_ms = sample.video_offset_ms - result[-1].video_offset_ms
        max_step = MAX_MOVE_PX_PER_SECOND * delta_ms / 1000
        clamped = min(max(sample.target_center_x, prev_x - max_step), prev_x + max_step)
        result.append(
            TargetSample(
                video_offset_ms=sample.video_offset_ms,
                target_center_x=clamped,
                target_type=sample.target_type,
                confidence=sample.confidence,
            )
        )
    return result


def smooth(samples: list[TargetSample]) -> list[TargetSample]:
    """중심 이동 평균(창 5)으로 잔여 흔들림을 제거한다.

    이동 평균은 인접 차이를 늘리지 않으므로 속도 제한 결과를 위반하지 않는다.
    """
    if len(samples) <= 2:
        return list(samples)

    xs = [s.target_center_x for s in samples]
    half = _SMOOTH_WINDOW // 2
    result: list[TargetSample] = []
    for i, sample in enumerate(samples):
        window = xs[max(0, i - half) : i + half + 1]
        result.append(
            TargetSample(
                video_offset_ms=sample.video_offset_ms,
                target_center_x=sum(window) / len(window),
                target_type=sample.target_type,
                confidence=sample.confidence,
            )
        )
    return result


def apply_dead_zone(samples: list[TargetSample]) -> list[TargetSample]:
    """데드존 — 공(target)이 크롭 중심 ±_DEAD_ZONE_HALF 안이면 크롭을 고정한다.

    벗어나면 공을 데드존 경계에 걸치도록만 크롭 중심을 이동한다(그만큼만 따라감).
    작은 흔들림·드리블에 크롭이 반응하지 않아 화면이 안정적으로 유지된다.

    뒤이은 limit_speed가 경계 이동을 최대 속도 이내로 캡하므로 속도 정책을 지킨다.
    """
    if not samples:
        return []

    result: list[TargetSample] = [samples[0]]
    crop_center = samples[0].target_center_x
    for sample in samples[1:]:
        delta = sample.target_center_x - crop_center
        if abs(delta) > _DEAD_ZONE_HALF:
            # 데드존을 벗어난 만큼만 이동 — 공을 경계에 걸침
            crop_center = sample.target_center_x - (
                _DEAD_ZONE_HALF if delta > 0 else -_DEAD_ZONE_HALF
            )
        # else: crop_center 유지(고정)
        result.append(
            TargetSample(
                video_offset_ms=sample.video_offset_ms,
                target_center_x=crop_center,
                target_type=sample.target_type,
                confidence=sample.confidence,
            )
        )
    return result


def stabilize(samples: list[TargetSample]) -> list[TargetSample]:
    """안정화 전체 파이프라인 — Outlier 제거 → 보간 → 데드존 → 속도 제한 → 평활화."""
    return smooth(limit_speed(apply_dead_zone(interpolate_gaps(remove_outliers(samples)))))

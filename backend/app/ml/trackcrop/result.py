"""Crop 좌표 결과 조립 및 자체 검증.

원본(LivePick)의 plan_builder + validate_crop_plan에서 Job/스키마 계약을 걷어내고
순수 좌표 결과(CropResult)만 만들도록 재구성했다.
"""

import math

from . import constants
from .types import CropResult, Keyframe, TargetSample, TargetType


def build_crop_result(
    samples: list[TargetSample],
    keyframes: list[Keyframe],
    *,
    duration_ms: int | None = None,
    source_width: int = constants.SOURCE_WIDTH,
    source_height: int = constants.SOURCE_HEIGHT,
    debug: list | None = None,
) -> CropResult:
    """CropResult를 생성한다.

    coverage 정의 (Sample 단위 비율):
    - ball_tracking_coverage: target_type ∈ {ball, ball_player}
    - player_tracking_coverage: target_type ∈ {ball_player, player_group}
    - fallback_coverage: target_type == center

    duration_ms 미지정 시 마지막 Sample의 offset을 Duration으로 본다.
    마지막 Keyframe이 Duration과 다르면 Duration으로 이동시킨다
    (검증 규칙 "마지막 Keyframe == Duration" 충족).
    """
    if duration_ms is None:
        duration_ms = samples[-1].video_offset_ms if samples else 0

    adjusted = list(keyframes)
    if adjusted and adjusted[-1].video_offset_ms != duration_ms:
        last = adjusted[-1]
        adjusted[-1] = Keyframe(
            video_offset_ms=duration_ms,
            x=last.x,
            target_type=last.target_type,
            confidence=last.confidence,
        )

    n = len(samples)
    ball = sum(
        1 for s in samples if s.target_type in (TargetType.BALL, TargetType.BALL_PLAYER)
    )
    player = sum(
        1 for s in samples if s.target_type in (TargetType.BALL_PLAYER, TargetType.PLAYER_GROUP)
    )
    fallback = sum(1 for s in samples if s.target_type == TargetType.CENTER)

    return CropResult(
        source_width=source_width,
        source_height=source_height,
        duration_ms=duration_ms,
        crop_width=constants.CROP_WIDTH,
        crop_height=constants.CROP_HEIGHT,
        crop_y=constants.CROP_Y,
        keyframes=adjusted,
        samples=list(samples),
        summary={
            "keyframe_count": len(adjusted),
            "ball_tracking_coverage": round(ball / n, 4) if n else 0.0,
            "player_tracking_coverage": round(player / n, 4) if n else 0.0,
            "fallback_coverage": round(fallback / n, 4) if n else 0.0,
        },
        debug=debug or [],
    )


def validate_crop_result(
    result: CropResult,
    max_move_px_per_second: float = constants.MAX_MOVE_PX_PER_SECOND,
) -> list[str]:
    """결과 자체 검증. 위반 목록을 반환한다 (빈 리스트 = 통과).

      1. Crop 규격 (width/height/y)이 constants와 일치
      2. Keyframe 최소 2개
      3. 첫 Keyframe video_offset_ms == 0
      4. 마지막 Keyframe == duration_ms
      5. Keyframe 시간 오름차순 · 중복 없음
      6. 모든 X가 0..CROP_X_MAX 범위
      7. X가 정수 Pixel
      8. 최대 이동 속도(1200px/s) 정책 충족
      9. NaN · 무한대 값 없음
    """
    violations: list[str] = []
    kfs = result.keyframes

    if (result.crop_width, result.crop_height, result.crop_y) != (
        constants.CROP_WIDTH,
        constants.CROP_HEIGHT,
        constants.CROP_Y,
    ):
        violations.append(
            f"crop 규격 {result.crop_width}×{result.crop_height}·Y{result.crop_y} "
            f"≠ {constants.CROP_WIDTH}×{constants.CROP_HEIGHT}·Y{constants.CROP_Y}"
        )
    if len(kfs) < 2:
        violations.append(f"Keyframe {len(kfs)}개 — 최소 2개 필요")
        return violations  # 이하 규칙은 Keyframe 2개 이상 전제

    if kfs[0].video_offset_ms != 0:
        violations.append(f"첫 Keyframe {kfs[0].video_offset_ms}ms ≠ 0")
    if kfs[-1].video_offset_ms != result.duration_ms:
        violations.append(
            f"마지막 Keyframe {kfs[-1].video_offset_ms}ms ≠ Duration {result.duration_ms}ms"
        )

    offsets = [k.video_offset_ms for k in kfs]
    if any(b <= a for a, b in zip(offsets, offsets[1:], strict=False)):
        violations.append("Keyframe 시간이 오름차순이 아니거나 중복 존재")

    if any(k.x < constants.CROP_X_MIN or k.x > constants.CROP_X_MAX for k in kfs):
        violations.append(f"X가 {constants.CROP_X_MIN}..{constants.CROP_X_MAX} 범위를 벗어남")
    if any(not isinstance(k.x, int) for k in kfs):
        violations.append("정수가 아닌 X 존재")

    for prev, nxt in zip(kfs, kfs[1:], strict=False):
        delta_ms = nxt.video_offset_ms - prev.video_offset_ms
        if delta_ms <= 0:
            continue  # 순서 위반은 위에서 보고됨
        max_move = max_move_px_per_second * delta_ms / 1000
        if abs(nxt.x - prev.x) > max_move:
            violations.append(
                f"{prev.video_offset_ms}→{nxt.video_offset_ms}ms 이동 "
                f"{abs(nxt.x - prev.x)}px가 최대 속도({max_move_px_per_second:g}px/s) 초과"
            )
            break

    numbers = [kf.confidence for kf in kfs] + [float(kf.x) for kf in kfs]
    if any(math.isnan(v) or math.isinf(v) for v in numbers):
        violations.append("NaN 또는 무한대 값 존재")

    return violations

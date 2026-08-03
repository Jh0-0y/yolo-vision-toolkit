"""전역 크롭 경로 최적화 (2-패스 설계의 패스 2 후반).

데드존 → 속도 제한 → 이동평균 체인을 목적함수 하나로 통합한다:

  J(c) = Σ w_follow·conf·max(|c−b| − dz, 0)²   (데드존 밖 이탈 페널티)
       + Σ w_inside·conf·(c−b)²                (데드존 안 미세 인력)
       + Σ w_vel·(Δc)² + Σ w_acc·(Δ²c)²        (팬 속도·가속 페널티)

IRLS(반복 재가중 최소제곱)로 푼다: 이탈 항은 데드존 경계까지의 거리에 대한
2차식이므로, 활성 집합을 갱신하며 선형계를 반복해서 풀면 수렴한다.
클립 전체를 한 번에 풀기 때문에 "곧 이동할 방향으로 미리 천천히 출발"이 나온다.

마지막에 크롭 범위 클램프 + 샘플 간 최대 이동량 하드 캡으로 스펙을 보장한다.
"""

import numpy as np

from .balltrack import ClipPlanConfig
from .constants import CROP_WIDTH, SOURCE_WIDTH
from .types import TargetSample

# 크롭 중심 허용 범위 (크롭 박스가 소스를 벗어나지 않게)
_CENTER_MIN = CROP_WIDTH / 2  # 304
_CENTER_MAX = SOURCE_WIDTH - CROP_WIDTH / 2  # 1616


def _difference_penalty(n: int, w_vel: float, w_acc: float) -> np.ndarray:
    """속도(1차 차분)·가속(2차 차분) 페널티의 정규방정식 행렬 (n×n)."""
    penalty = np.zeros((n, n))
    if n >= 2 and w_vel > 0:
        d1 = np.diff(np.eye(n), axis=0)
        penalty += w_vel * d1.T @ d1
    if n >= 3 and w_acc > 0:
        d2 = np.diff(np.eye(n), n=2, axis=0)
        penalty += w_acc * d2.T @ d2
    return penalty


def optimize_path(samples: list[TargetSample], cfg: ClipPlanConfig | None = None) -> list[float]:
    """타깃 시퀀스에서 최적 크롭 중심 경로를 계산한다 (같은 그리드, 같은 길이)."""
    cfg = cfg or ClipPlanConfig()
    n = len(samples)
    if n == 0:
        return []
    b = np.array([s.target_center_x for s in samples], dtype=float)
    if n < 3:
        return list(np.clip(b, _CENTER_MIN, _CENTER_MAX))

    conf = np.array([max(s.confidence, cfg.min_follow_conf) for s in samples])
    dz = cfg.dead_zone_half
    smooth = _difference_penalty(n, cfg.w_vel, cfg.w_acc)

    c = b.copy()
    for _ in range(cfg.irls_iters):
        residual = c - b
        outside = np.abs(residual) > dz
        # 이탈 항의 인력 지점: 데드존 밖이면 경계(b ± dz), 안이면 b(약한 인력)
        pull = np.where(outside, b + np.sign(residual) * dz, b)
        weight = conf * np.where(outside, cfg.w_follow, cfg.w_inside)
        a_matrix = np.diag(weight) + smooth
        c = np.linalg.solve(a_matrix, weight * pull)
        c = np.clip(c, _CENTER_MIN, _CENTER_MAX)

    return list(_cap_speed(c, samples, cfg))


def _cap_speed(path: np.ndarray, samples: list[TargetSample], cfg: ClipPlanConfig) -> np.ndarray:
    """샘플 간 이동량을 스펙 최대 속도로 하드 캡 (앞→뒤 한 번)."""
    capped = path.copy()
    for i in range(1, len(capped)):
        dt_ms = samples[i].video_offset_ms - samples[i - 1].video_offset_ms
        max_step = cfg.max_move_px_per_second * max(dt_ms, 0) / 1000
        delta = capped[i] - capped[i - 1]
        if abs(delta) > max_step:
            capped[i] = capped[i - 1] + np.sign(delta) * max_step
    return capped

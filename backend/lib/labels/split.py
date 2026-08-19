"""train / val / test 배정 — 순수 계산. 파일도 잡도 모른다.

**기본은 아직 배정되지 않은 것만 나눈다.** 이미 train/val/test 에 들어간 이미지는
건드리지 않는다 — 안 그러면 어제 test 였던 이미지가 오늘 train 으로 옮겨 가면서
학습에 새어 들어가고, 이전 평가와 비교가 성립하지 않는다.

전부 다시 섞는 것은 호출자가 `reassign_all=True` 로 **명시**해야 한다.

시드를 받으므로 같은 입력·같은 시드면 같은 결과가 나온다.
"""

from __future__ import annotations

import random

SPLITS = ("train", "val", "test")


class SplitError(Exception):
    """비율이 배정에 쓸 수 없는 값이다."""


def normalize_ratios(train: float, val: float, test: float) -> tuple[float, float, float]:
    """합이 1이 되도록 맞춘다. `8:1:1` 처럼 정수 비로 줘도 된다."""
    total = train + val + test
    if total <= 0 or min(train, val, test) < 0:
        raise SplitError("Ratios must be non-negative and add up to more than zero")
    return train / total, val / total, test / total


def assign(
    stems: list[str],
    current: dict[str, str],
    *,
    train: float = 8,
    val: float = 1,
    test: float = 1,
    seed: int = 0,
    reassign_all: bool = False,
) -> dict[str, str]:
    """`stems` 에 대한 새 배정표를 돌려준다.

    `current` 는 지금 배정이다. `reassign_all` 이 아니면 거기 있는 것은 그대로 두고
    **나머지만** 나눈다. `stems` 에 없는 항목은 결과에서 빠진다(이미지가 지워진 것).

    나누는 방식은 "섞어서 앞에서부터 잘라 담기"다. 반올림으로 한두 장이 남으면
    **train 에 몰아준다** — 학습 데이터가 가장 많아야 하고, val/test 가 의도보다
    한 장 적은 것이 train 이 한 장 적은 것보다 낫다.
    """
    r_train, r_val, r_test = normalize_ratios(train, val, test)

    keep: dict[str, str] = {}
    if not reassign_all:
        keep = {s: current[s] for s in stems if current.get(s) in SPLITS}

    pool = [s for s in stems if s not in keep]
    if not pool:
        return keep

    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * r_val)
    n_test = int(n * r_test)
    # 아주 작은 풀에서도 val/test 가 통째로 비지 않게 한 장은 준다
    if n >= 3:
        if r_val > 0 and n_val == 0:
            n_val = 1
        if r_test > 0 and n_test == 0:
            n_test = 1
    n_train = n - n_val - n_test
    if n_train < 0:  # 비율이 극단적일 때 — train 을 0 밑으로 내리지 않는다
        n_train, n_val, n_test = 0, min(n_val, n), n - min(n_val, n)

    out = dict(keep)
    for i, stem in enumerate(shuffled):
        if i < n_train:
            out[stem] = "train"
        elif i < n_train + n_val:
            out[stem] = "val"
        else:
            out[stem] = "test"
    return out


def counts(splits: dict[str, str]) -> dict[str, int]:
    out = {s: 0 for s in SPLITS}
    for v in splits.values():
        if v in out:
            out[v] += 1
    return out

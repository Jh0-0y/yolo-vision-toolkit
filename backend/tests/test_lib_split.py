"""train/val/test 배정 규칙.

가장 중요한 것 하나: **이미 배정된 것은 안 건드린다.** 이미지를 더 넣고 다시 나눠도
어제 test 였던 것은 오늘도 test 여야 한다 — 안 그러면 test 가 학습에 새어 들어가
이전 평가와 비교가 성립하지 않는다.
"""

import pytest

from lib.labels.split import SplitError, assign, counts, normalize_ratios


def test_splits_an_empty_board_by_ratio():
    stems = [f"img{i}" for i in range(100)]

    out = assign(stems, {}, train=8, val=1, test=1, seed=1)

    assert counts(out) == {"train": 80, "val": 10, "test": 10}


def test_existing_assignments_are_never_touched():
    """이 계획서의 핵심 규칙 — 나중에 이미지를 더해도 기존 배정은 그대로다."""
    old = {f"img{i}": "test" for i in range(10)}
    stems = [f"img{i}" for i in range(30)]

    out = assign(stems, old, seed=1)

    for stem, split in old.items():
        assert out[stem] == split, f"{stem} moved out of {split}"
    assert counts(out)["test"] >= 10


def test_only_the_unassigned_get_new_homes():
    old = {"a": "train", "b": "val"}

    out = assign(["a", "b", "c", "d", "e"], old, seed=1)

    assert (out["a"], out["b"]) == ("train", "val")
    assert set(out) == {"a", "b", "c", "d", "e"}


def test_reassign_all_reshuffles_everything():
    """전부 다시 섞는 것은 명시했을 때만 일어난다."""
    old = {f"img{i}": "test" for i in range(30)}
    stems = list(old)

    out = assign(stems, old, train=8, val=1, test=1, seed=1, reassign_all=True)

    assert counts(out)["train"] > 0  # 더 이상 전부 test 가 아니다


def test_deleted_images_drop_out_of_the_table():
    """이미지를 지우면 배정표에 남은 줄이 따라 나오지 않는다."""
    out = assign(["a"], {"a": "train", "gone": "test"}, seed=1)

    assert set(out) == {"a"}


def test_the_same_seed_gives_the_same_split():
    stems = [f"img{i}" for i in range(50)]

    assert assign(stems, {}, seed=7) == assign(stems, {}, seed=7)


def test_a_different_seed_gives_a_different_split():
    stems = [f"img{i}" for i in range(50)]

    assert assign(stems, {}, seed=1) != assign(stems, {}, seed=2)


def test_rounding_leftovers_go_to_train():
    """반올림으로 남는 장은 train 이 가져간다 — 학습 데이터가 가장 많아야 한다."""
    out = counts(assign([f"i{n}" for n in range(7)], {}, train=8, val=1, test=1, seed=1))

    assert out["train"] == 5 and out["val"] == 1 and out["test"] == 1
    assert sum(out.values()) == 7


def test_a_tiny_pool_still_fills_val_and_test():
    out = counts(assign(["a", "b", "c"], {}, train=8, val=1, test=1, seed=1))

    assert out == {"train": 1, "val": 1, "test": 1}


def test_one_image_goes_to_train():
    assert counts(assign(["only"], {}, seed=1)) == {"train": 1, "val": 0, "test": 0}


def test_a_zero_ratio_split_is_left_empty():
    out = counts(assign([f"i{n}" for n in range(20)], {}, train=1, val=0, test=0, seed=1))

    assert out == {"train": 20, "val": 0, "test": 0}


def test_ratios_are_normalized():
    assert normalize_ratios(8, 1, 1) == pytest.approx((0.8, 0.1, 0.1))
    assert normalize_ratios(0.8, 0.1, 0.1) == pytest.approx((0.8, 0.1, 0.1))


@pytest.mark.parametrize("bad", [(0, 0, 0), (-1, 1, 1)])
def test_nonsense_ratios_are_rejected(bad):
    with pytest.raises(SplitError):
        normalize_ratios(*bad)


def test_unknown_current_values_are_treated_as_unassigned():
    """splits.json 에 모르는 값이 들어 있어도 배정이 막히지 않는다."""
    out = assign(["a", "b", "c"], {"a": "holdout"}, seed=1)

    assert out["a"] in ("train", "val", "test")

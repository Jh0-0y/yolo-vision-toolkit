"""연구실 크롭 런의 파일 규약 — 상태는 progress.jsonl 에서 파생되고, 목록은
디렉터리를 훑어 만들며, 원본 사본은 하드링크가 안 되면 복사로 떨어진다.
DB 도 인메모리 상태도 끼지 않는 부분이라 파일만 깔아 놓고 검증할 수 있다."""

import json

import pytest

from app.core.config import settings
from app.services import lab_crop_runs
from infra import jobs


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _emit(crop_id: str, event: dict) -> None:
    jobs.at(settings.jobs_dir, crop_id).ensure().emit(event)


def _create(lab_id: str = "lab_1", **settings_override) -> str:
    return lab_crop_runs.create(
        lab_id,
        "test run",
        {
            "source_video_id": "vid_1",
            "source_name": "clip.mp4",
            "models": [{"model_id": "m_1", "name": "yolo26n", "mode": "full"}],
            "overrides": {"ball_weight": 0.7},
            "crop_w": 608,
            "crop_h": 1080,
            "toggles": {"crop_box": True},
            **settings_override,
        },
    )


# ---------- 상태 ----------


def test_status_is_running_until_a_terminal_event(data_dir):
    crop_id = _create()
    assert lab_crop_runs.status_of(crop_id) == ("running", None)

    _emit(crop_id, {"phase": "detect", "done": 10})
    assert lab_crop_runs.status_of(crop_id) == ("running", None)

    _emit(crop_id, {"phase": "done"})
    assert lab_crop_runs.status_of(crop_id) == ("done", None)


def test_status_carries_the_error_message(data_dir):
    crop_id = _create()
    _emit(crop_id, {"phase": "error", "msg": "boom"})
    assert lab_crop_runs.status_of(crop_id) == ("error", "boom")


def test_failed_runs_stay_in_the_list(data_dir):
    """실패한 시도도 남아야 어떤 설정이 안 되는지가 기록된다."""
    crop_id = _create()
    _emit(crop_id, {"phase": "error", "msg": "no detections"})

    (run,) = lab_crop_runs.list_runs("lab_1")
    assert (run["id"], run["status"], run["error"]) == (crop_id, "error", "no detections")


def test_reconcile_fails_runs_that_a_restart_killed(data_dir):
    running = _create()
    finished = _create()
    _emit(finished, {"phase": "done"})

    lab_crop_runs.reconcile_on_boot()

    assert lab_crop_runs.status_of(running)[0] == "error"
    assert lab_crop_runs.status_of(finished)[0] == "done"


# ---------- 목록 ----------


def test_list_reads_the_summary_out_of_crop_json(data_dir):
    crop_id = _create()
    lab_crop_runs.artifact("lab_1", crop_id, lab_crop_runs.CROP_NAME).write_text(
        json.dumps({"summary": {"keyframeCount": 12}})
    )
    _emit(crop_id, {"phase": "done"})

    (run,) = lab_crop_runs.list_runs("lab_1")
    assert run["status"] == "done"
    assert run["summary"] == {"keyframeCount": 12}
    assert run["overrides"] == {"ball_weight": 0.7}
    assert (run["crop_w"], run["crop_h"]) == (608, 1080)
    assert run["has_crop_json"] is True
    assert (run["has_wide"], run["has_cut"]) == (False, False)


def test_size_counts_every_artifact(data_dir):
    crop_id = _create()
    for name, body in (
        (lab_crop_runs.CROP_NAME, b"12"),
        (lab_crop_runs.WIDE_NAME, b"1234"),
        (lab_crop_runs.CUT_NAME, b"123456"),
    ):
        lab_crop_runs.artifact("lab_1", crop_id, name).write_bytes(body)

    (run,) = lab_crop_runs.list_runs("lab_1")
    assert run["size_bytes"] == 12
    assert (run["has_wide"], run["has_cut"]) == (True, True)


def test_runs_of_other_labs_are_not_mixed_in(data_dir):
    mine = _create("lab_1")
    _create("lab_2")

    assert [r["id"] for r in lab_crop_runs.list_runs("lab_1")] == [mine]


def test_list_of_an_unknown_lab_is_empty(data_dir):
    assert lab_crop_runs.list_runs("nope") == []


def test_delete_removes_the_run_and_its_progress(data_dir):
    crop_id = _create()
    _emit(crop_id, {"phase": "done"})

    lab_crop_runs.delete("lab_1", crop_id)

    assert lab_crop_runs.list_runs("lab_1") == []
    assert not (settings.jobs_dir / crop_id).exists()


# ---------- 그리기 토글 ----------


def test_no_toggle_means_no_overlay(data_dir):
    assert lab_crop_runs.wants_overlay({}) is False
    assert lab_crop_runs.wants_overlay({k: False for k in lab_crop_runs.DRAW_TOGGLES}) is False


@pytest.mark.parametrize("toggle", lab_crop_runs.DRAW_TOGGLES)
def test_any_single_toggle_turns_the_overlay_on(data_dir, toggle):
    assert lab_crop_runs.wants_overlay({toggle: True}) is True


def test_unknown_keys_do_not_turn_the_overlay_on(data_dir):
    """구버전 프론트가 남긴 키 하나가 원본 사본을 렌더로 바꾸면 안 된다."""
    assert lab_crop_runs.wants_overlay({"show_everything": True}) is False


# ---------- 가로 영상의 정체 ----------


def test_wide_kind_is_empty_until_the_video_exists(data_dir):
    crop_id = _create()
    assert lab_crop_runs.wide_kind("lab_1", crop_id, {"toggles": {}}) == ""


def test_wide_kind_is_render_when_anything_was_drawn(data_dir):
    crop_id = _create()
    lab_crop_runs.artifact("lab_1", crop_id, lab_crop_runs.WIDE_NAME).write_bytes(b"x")
    meta = {"toggles": {"crop_box": True}}

    assert lab_crop_runs.wide_kind("lab_1", crop_id, meta) == "render"


def test_wide_kind_reads_the_link_count(data_dir, tmp_path):
    """하드링크인지는 파일이 이미 알고 있다 — 따로 적어 두지 않는다."""
    crop_id = _create()
    src = tmp_path / "src.mp4"
    src.write_bytes(b"raw")
    lab_crop_runs.link_or_copy(src, lab_crop_runs.artifact("lab_1", crop_id, lab_crop_runs.WIDE_NAME))

    assert lab_crop_runs.wide_kind("lab_1", crop_id, {"toggles": {}}) == "link"


def test_wide_kind_is_copy_for_a_standalone_file(data_dir):
    crop_id = _create()
    lab_crop_runs.artifact("lab_1", crop_id, lab_crop_runs.WIDE_NAME).write_bytes(b"x")

    assert lab_crop_runs.wide_kind("lab_1", crop_id, {"toggles": {}}) == "copy"


def test_the_run_row_carries_the_derived_kind(data_dir, tmp_path):
    crop_id = _create(toggles={})
    src = tmp_path / "src.mp4"
    src.write_bytes(b"raw")
    lab_crop_runs.link_or_copy(src, lab_crop_runs.artifact("lab_1", crop_id, lab_crop_runs.WIDE_NAME))

    (run,) = lab_crop_runs.list_runs("lab_1")
    assert run["wide_kind"] == "link"


# ---------- 원본 사본: 하드링크 · 폴백 ----------


def test_link_or_copy_hardlinks_within_the_same_volume(data_dir, tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"raw")
    dst = tmp_path / "run" / "wide.mp4"

    assert lab_crop_runs.link_or_copy(src, dst) == "link"
    assert dst.read_bytes() == b"raw"
    assert src.stat().st_ino == dst.stat().st_ino  # 한 벌만 쓴다


def test_the_link_survives_deleting_the_original(data_dir, tmp_path):
    """아카이브에서 영상을 지워도 런은 온전해야 한다."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"raw")
    dst = tmp_path / "run" / "wide.mp4"
    lab_crop_runs.link_or_copy(src, dst)

    src.unlink()

    assert dst.exists() and dst.read_bytes() == b"raw"


def test_link_or_copy_falls_back_to_copying(data_dir, tmp_path, monkeypatch):
    """다른 볼륨이면 os.link 가 OSError 다 — 용량을 쓰더라도 복사한다."""
    import os

    src = tmp_path / "src.mp4"
    src.write_bytes(b"raw")
    dst = tmp_path / "run" / "wide.mp4"
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError("xdev")))

    assert lab_crop_runs.link_or_copy(src, dst) == "copy"
    assert dst.read_bytes() == b"raw"
    assert src.stat().st_ino != dst.stat().st_ino


def test_link_or_copy_replaces_an_existing_target(data_dir, tmp_path):
    """다시 돌린 런이 옛 사본 위에 얹히지 않게."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"new")
    dst = tmp_path / "run" / "wide.mp4"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"stale")

    lab_crop_runs.link_or_copy(src, dst)

    assert dst.read_bytes() == b"new"


@pytest.mark.parametrize("bad", ["", ".", "../secrets", "a/b", "a\\b"])
def test_invalid_ids_are_rejected(bad):
    assert lab_crop_runs.valid_id(bad) is False

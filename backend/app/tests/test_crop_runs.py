"""크롭 런의 파일 규약 — 상태는 progress.jsonl 에서 파생되고, 목록은 디렉터리를
훑어 만들며, 영상만 만료된다. DB 도 인메모리 상태도 끼지 않는 부분이라 파일만
깔아 놓고 검증할 수 있다."""

import json
import os
import time

import pytest

from app.core.config import settings
from app.services import crop_runs
from infra import jobs


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _emit(crop_id: str, event: dict) -> None:
    jobs.at(settings.jobs_dir, crop_id).ensure().emit(event)


def test_status_is_running_until_a_terminal_event(data_dir):
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {})
    assert crop_runs.status_of(crop_id) == ("running", None)

    _emit(crop_id, {"phase": "crop_analyze"})
    assert crop_runs.status_of(crop_id) == ("running", None)

    _emit(crop_id, {"phase": "done"})
    assert crop_runs.status_of(crop_id) == ("done", None)


def test_status_carries_the_error_message(data_dir):
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {})
    _emit(crop_id, {"phase": "error", "msg": "boom"})
    assert crop_runs.status_of(crop_id) == ("error", "boom")


def test_list_reads_the_summary_out_of_crop_json(data_dir):
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {"overrides": {"ball_weight": 2}})
    (crop_runs.run_dir("p1", crop_id) / crop_runs.CROP_NAME).write_text(
        json.dumps({"summary": {"keyframeCount": 12}})
    )
    _emit(crop_id, {"phase": "done"})

    (run,) = crop_runs.list_runs("p1")
    assert run["id"] == crop_id
    assert run["status"] == "done"
    assert run["summary"] == {"keyframeCount": 12}
    assert run["settings"] == {"overrides": {"ball_weight": 2}}
    assert run["has_crop_json"] is True
    assert run["has_video"] is False


def test_failed_runs_stay_in_the_list(data_dir):
    """실패한 시도도 남아야 어떤 설정이 안 되는지가 기록된다."""
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {})
    _emit(crop_id, {"phase": "error", "msg": "no detections"})

    (run,) = crop_runs.list_runs("p1")
    assert (run["id"], run["status"], run["error"]) == (crop_id, "error", "no detections")


def test_list_is_newest_first(data_dir):
    first = crop_runs.create("p1", "json", "a.mp4", {})
    time.sleep(0.01)
    second = crop_runs.create("p1", "json", "b.mp4", {})
    assert [r["id"] for r in crop_runs.list_runs("p1")] == [second, first]


def test_sweep_drops_the_source_of_a_finished_run_but_keeps_a_fresh_video(data_dir):
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {})
    run = crop_runs.run_dir("p1", crop_id)
    (run / f"{crop_runs.SOURCE_STEM}.mp4").write_bytes(b"raw")
    (run / crop_runs.VIDEO_NAME).write_bytes(b"rendered")
    _emit(crop_id, {"phase": "done"})

    crop_runs.sweep_expired()

    assert not (run / f"{crop_runs.SOURCE_STEM}.mp4").exists()
    assert (run / crop_runs.VIDEO_NAME).exists()
    assert crop_runs.read_meta("p1", crop_id).get("video_expired") is None


def test_sweep_expires_an_aged_video_and_marks_the_meta(data_dir):
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {})
    run = crop_runs.run_dir("p1", crop_id)
    video = run / crop_runs.VIDEO_NAME
    video.write_bytes(b"rendered")
    aged = time.time() - crop_runs.VIDEO_TTL_SEC - 60
    os.utime(video, (aged, aged))
    _emit(crop_id, {"phase": "done"})

    crop_runs.sweep_expired()

    assert not video.exists()
    run_out = crop_runs.to_out("p1", crop_id, crop_runs.read_meta("p1", crop_id))
    assert (run_out["video_expired"], run_out["has_video"]) == (True, False)


def test_sweep_leaves_a_running_run_alone(data_dir):
    """워커가 아직 쓰고 있는 파일이라 나이만 보고 지우면 안 된다."""
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {})
    run = crop_runs.run_dir("p1", crop_id)
    source = run / f"{crop_runs.SOURCE_STEM}.mp4"
    source.write_bytes(b"raw")
    video = run / crop_runs.VIDEO_NAME
    video.write_bytes(b"partial")
    aged = time.time() - crop_runs.VIDEO_TTL_SEC - 60
    os.utime(video, (aged, aged))

    crop_runs.sweep_expired()

    assert source.exists()
    assert video.exists()


def test_reconcile_fails_runs_that_a_restart_killed(data_dir):
    running = crop_runs.create("p1", "json", "a.mp4", {})
    finished = crop_runs.create("p1", "json", "b.mp4", {})
    _emit(finished, {"phase": "done"})

    crop_runs.reconcile_on_boot()

    assert crop_runs.status_of(running)[0] == "error"
    assert crop_runs.status_of(finished)[0] == "done"


def test_delete_removes_the_run_and_its_progress(data_dir):
    crop_id = crop_runs.create("p1", "json", "clip.mp4", {})
    _emit(crop_id, {"phase": "done"})

    crop_runs.delete("p1", crop_id)

    assert crop_runs.list_runs("p1") == []
    assert not (settings.jobs_dir / crop_id).exists()


@pytest.mark.parametrize("bad", ["", ".", "../secrets", "a/b", "a\\b"])
def test_invalid_ids_are_rejected(bad):
    assert crop_runs.valid_id(bad) is False

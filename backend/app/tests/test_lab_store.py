"""연구실의 파일 규약 — 영상은 사이드카가 확장자를 들고, 목록은 디렉터리를 훑어
만들며, 원본을 지워도 크롭 런 수는 런 쪽 기록에서 나온다. DB 가 끼지 않는 부분이라
파일만 깔아 놓고 검증할 수 있다."""

import json

import pytest

from app.core.config import settings
from app.services import lab_store


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _add_video(lab_id: str, video_id: str, ext: str = ".mp4", **extra) -> None:
    lab_store.ensure_dirs(lab_id)
    (lab_store.videos_dir(lab_id) / f"{video_id}{ext}").write_bytes(b"raw")
    meta = lab_store.new_video_meta(video_id, f"{video_id}{ext}", ext, {"width": 1920})
    lab_store.write_video_meta(lab_id, video_id, {**meta, **extra})


def _add_run(lab_id: str, crop_id: str, source_video_id: str) -> None:
    run = lab_store.crops_dir(lab_id) / crop_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(json.dumps({"source_video_id": source_video_id}))


def test_ensure_dirs_lays_out_the_lab(data_dir):
    lab_store.ensure_dirs("lab_1")
    assert lab_store.videos_dir("lab_1").is_dir()
    assert lab_store.crops_dir("lab_1").is_dir()


def test_marker_names_the_directory(data_dir):
    lab_store.ensure_dirs("lab_1")
    lab_store.write_marker("lab_1", "basketball")
    marker = json.loads((lab_store.lab_dir("lab_1") / "lab.json").read_text())
    assert marker == {"id": "lab_1", "name": "basketball"}


def test_video_file_uses_the_extension_from_the_sidecar(data_dir):
    """컨테이너가 여러 가지라 확장자를 추측하지 않는다 — 사이드카가 진실이다."""
    _add_video("lab_1", "vid_1", ext=".mkv")
    path = lab_store.video_file("lab_1", "vid_1")
    assert path is not None and path.name == "vid_1.mkv"


def test_video_file_is_none_when_the_sidecar_is_missing(data_dir):
    lab_store.ensure_dirs("lab_1")
    (lab_store.videos_dir("lab_1") / "vid_1.mp4").write_bytes(b"raw")
    assert lab_store.video_file("lab_1", "vid_1") is None


def test_list_videos_is_newest_first_and_carries_size(data_dir):
    _add_video("lab_1", "vid_old", created_at="2026-01-01T00:00:00+00:00")
    _add_video("lab_1", "vid_new", created_at="2026-08-01T00:00:00+00:00")

    videos = lab_store.list_videos("lab_1")

    assert [v["id"] for v in videos] == ["vid_new", "vid_old"]
    assert all(v["size_bytes"] == 3 for v in videos)


def test_list_videos_of_an_unknown_lab_is_empty(data_dir):
    assert lab_store.list_videos("nope") == []


def test_probe_meta_reads_the_geometry(data_dir, monkeypatch):
    """업로드가 사이드카에 넣을 값. `lib.video.probe` 는 **함수**라 그대로 부른다 —
    모듈처럼 한 번 더 파고들면 업로드가 500 이 된다(실제로 그랬다)."""
    from types import SimpleNamespace

    import lib.video

    monkeypatch.setattr(
        lib.video,
        "probe",
        lambda p: SimpleNamespace(width=1920, height=1080, fps=29.97, duration_ms=5005),
    )

    assert lab_store.probe_meta(data_dir / "clip.mp4") == {
        "width": 1920,
        "height": 1080,
        "fps": 29.97,
        "duration_ms": 5005,
    }


def test_probe_meta_falls_back_to_zeros_when_the_file_will_not_open(data_dir, monkeypatch):
    """목록에 못 뜨는 것보다 낫다 — 실제 실패는 크롭을 돌릴 때 드러난다."""
    import lib.video

    def _boom(_path):
        raise ValueError("not a video")

    monkeypatch.setattr(lib.video, "probe", _boom)

    assert lab_store.probe_meta(data_dir / "junk.mp4") == {
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration_ms": 0,
    }


def test_delete_video_drops_the_file_and_the_sidecar(data_dir):
    _add_video("lab_1", "vid_1")

    lab_store.delete_video("lab_1", "vid_1")

    assert not (lab_store.videos_dir("lab_1") / "vid_1.mp4").exists()
    assert not lab_store.video_meta_path("lab_1", "vid_1").exists()


def test_delete_video_leaves_crop_runs_alone(data_dir):
    """런은 원본을 참조하지 않고 자기 사본을 갖는다 — 아카이브에서 지워도 온전하다."""
    _add_video("lab_1", "vid_1")
    _add_run("lab_1", "crop_1", "vid_1")

    lab_store.delete_video("lab_1", "vid_1")

    assert (lab_store.crops_dir("lab_1") / "crop_1" / "run.json").exists()
    assert lab_store.count_runs("lab_1") == 1


def test_run_counts_are_read_from_the_run_metadata(data_dir):
    _add_run("lab_1", "crop_1", "vid_1")
    _add_run("lab_1", "crop_2", "vid_1")
    _add_run("lab_1", "crop_3", "vid_2")

    assert lab_store.count_runs("lab_1") == 3
    assert lab_store.count_runs_for_video("lab_1", "vid_1") == 2
    assert lab_store.count_runs_for_video("lab_1", "vid_2") == 1
    assert lab_store.count_runs_for_video("lab_1", "vid_missing") == 0


def test_a_broken_run_json_does_not_break_the_count(data_dir):
    lab_store.ensure_dirs("lab_1")
    broken = lab_store.crops_dir("lab_1") / "crop_bad"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{not json")
    _add_run("lab_1", "crop_ok", "vid_1")

    assert lab_store.count_runs_for_video("lab_1", "vid_1") == 1


def test_delete_lab_takes_videos_and_runs_with_it(data_dir):
    _add_video("lab_1", "vid_1")
    _add_run("lab_1", "crop_1", "vid_1")

    lab_store.delete_lab("lab_1")

    assert not lab_store.lab_dir("lab_1").exists()


@pytest.mark.parametrize("bad", ["", ".", "../secrets", "a/b", "a\\b"])
def test_invalid_ids_are_rejected(bad):
    assert lab_store.valid_id(bad) is False

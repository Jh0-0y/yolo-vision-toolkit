"""infra/jobs 계약 테스트 — 진행률 파일과 취소 센티널.

프로세스 둘이 파일로만 만나는 지점이라, 여기 계약이 깨지면 진행률이 멈추거나
취소가 안 먹는다. 워커 없이 파일만으로 검증한다.
"""

from __future__ import annotations

import json

from infra import jobs


def test_emit_appends_one_json_line_with_ts(tmp_path):
    job = jobs.at(tmp_path, "j1").prepare()

    job.emit({"phase": "start", "total": 3})
    job.emit({"phase": "done"})

    lines = job.progress_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["phase"] == "start"
    assert first["total"] == 3
    assert isinstance(first["ts"], float)  # ts 는 호출자가 아니라 emit 이 붙인다


def test_emit_keeps_non_ascii_readable(tmp_path):
    job = jobs.at(tmp_path, "j1").prepare()

    job.emit({"phase": "error", "msg": "한글 오류"})

    assert "한글 오류" in job.progress_path.read_text()


def test_read_resumes_from_offset(tmp_path):
    job = jobs.at(tmp_path, "j1").prepare()
    job.emit({"phase": "a"})

    first, offset = job.read()
    assert [e["phase"] for e in first] == ["a"]

    job.emit({"phase": "b"})
    second, offset2 = job.read(offset)

    assert [e["phase"] for e in second] == ["b"]  # 이미 읽은 건 다시 안 준다
    assert offset2 > offset


def test_read_holds_back_a_partial_trailing_line(tmp_path):
    """쓰는 쪽이 줄을 다 못 썼을 때 깨진 JSON 을 흘리지 않는다."""
    job = jobs.at(tmp_path, "j1").prepare()
    job.emit({"phase": "a"})
    with open(job.progress_path, "a") as f:
        f.write('{"phase": "half')  # 개행 없음 — 아직 쓰는 중

    events, offset = job.read()

    assert [e["phase"] for e in events] == ["a"]

    with open(job.progress_path, "a") as f:  # 나머지가 마저 쓰이면
        f.write('"}\n')
    rest, _ = job.read(offset)
    assert [e["phase"] for e in rest] == ["half"]


def test_read_missing_file_returns_empty(tmp_path):
    assert jobs.at(tmp_path, "nope").read(0) == ([], 0)


def test_cancel_roundtrip(tmp_path):
    job = jobs.at(tmp_path, "j1").prepare()
    assert not job.cancelled()

    job.request_cancel()

    assert job.cancelled()


def test_prepare_clears_a_stale_cancel(tmp_path):
    """잡 디렉터리를 재사용해도 이전 취소가 새 잡을 즉시 죽이면 안 된다."""
    job = jobs.at(tmp_path, "j1").prepare()
    job.request_cancel()

    jobs.at(tmp_path, "j1").prepare()

    assert not job.cancelled()


def test_prepare_keeps_existing_progress_but_reset_empties_it(tmp_path):
    job = jobs.at(tmp_path, "j1").prepare()
    job.emit({"phase": "a"})

    jobs.at(tmp_path, "j1").prepare()
    assert job.progress_path.read_text() != ""

    jobs.at(tmp_path, "j1").reset()
    assert job.progress_path.read_text() == ""

"""annotate 워커 골든 테스트 — 모드별 산출물을 못박아 둔다.

이 워커는 설정 조합에 따라 완전히 다른 것을 만든다(오버레이 영상 · 세로 컷 클립 ·
JSON만 · 추론 없는 컷). 리팩터링으로 렌더 경로를 합칠 때 **어느 조합도 조용히
달라지지 않았음**을 보증하는 것이 이 파일의 목적이다.

가짜로 두는 것은 **모델 경계 하나뿐**이다 — `ultralytics.YOLO` 와 검출 패스
(`build_detector`·`detect_video`). 크롭 좌표 계산(`plan_from_detections`)·궤적
보간·그리기·잘라내기·mp4v 쓰기·H.264 인코딩·취소·정리는 전부 진짜로 돈다.
그래야 "조립을 바꿔도 결과가 같다"를 확인할 수 있다.
"""

from __future__ import annotations

import json
import sys
import types

import cv2
import numpy as np
import pytest

import adaptive_crop
from adaptive_crop import Detection
from app.workers import annotate_worker
from lib import video
from lib.crop.geometry import crop_width_for

pytestmark = pytest.mark.skipif(
    not video.encode.shutil.which("ffmpeg"), reason="ffmpeg 없이는 산출물을 만들 수 없다"
)

W, H, FPS, FRAMES = 640, 360, 25.0, 25  # 1초짜리 최소 영상
CROP_W = crop_width_for(H, W)  # 202


# ---------- 소스 영상 ----------


def _make_video(path):
    """프레임마다 밝기가 달라 서로 구분되는 합성 영상."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FRAMES):
        frame = np.full((H, W, 3), (i * 7) % 200, dtype=np.uint8)
        cv2.rectangle(frame, (i * 20, 100), (i * 20 + 40, 160), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


# ---------- 모델 경계 가짜 ----------


def _detections(interval_ms=100):
    """공 하나가 왼쪽에서 오른쪽으로 지나가고 선수 둘이 따라간다."""
    out = []
    n = int(FRAMES / FPS * 1000 / interval_ms) + 1
    for k in range(n):
        ms = k * interval_ms
        x = 40 + (W - 120) * k / max(1, n - 1)
        out.append(
            (
                ms,
                [
                    Detection("ball", None, x, 150.0, 18.0, 18.0, 0.9, ms),
                    Detection("player", 1, x - 60, 120.0, 40.0, 110.0, 0.85, ms),
                    Detection("player", 2, x + 40, 130.0, 40.0, 105.0, 0.8, ms),
                ],
            )
        )
    return out


class _Tensorish:
    """`.cpu().numpy()` 만 흉내 낸다 — 워커가 torch 텐서에 쓰는 유일한 경로."""

    def __init__(self, arr):
        self._a = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, n, offset):
        self.xyxy = _Tensorish(
            np.array([[offset + i * 10, 100, offset + i * 10 + 30, 160] for i in range(n)],
                     dtype=float)
        )
        self.cls = _Tensorish(np.zeros(n, dtype=float))
        self.conf = _Tensorish(np.full(n, 0.9))
        self.id = _Tensorish(np.arange(1, n + 1, dtype=float))


class _FakeYOLO:
    """model.track(stream=True) 이 돌려주는 결과 스트림을 소스 영상에서 만든다."""

    def __init__(self, pt):
        self.pt = pt

    def track(self, source, **kw):
        cap = cv2.VideoCapture(str(source))
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield types.SimpleNamespace(
                    orig_img=frame, boxes=_Boxes(2, 20 + i * 8), names={0: "ball"}
                )
                i += 1
        finally:
            cap.release()


class _FakeYOLONoBoxes(_FakeYOLO):
    """검출이 하나도 없는 스트림 — 박스를 그리는지 대조하는 데 쓴다."""

    def track(self, source, **kw):
        for r in super().track(source, **kw):
            r.boxes = None
            yield r


def _patch_model(monkeypatch, yolo_cls=_FakeYOLO):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=yolo_cls))
    monkeypatch.setattr(adaptive_crop, "build_detector", lambda entries, device: object())
    monkeypatch.setattr(
        adaptive_crop,
        "detect_video",
        lambda src, **kw: _detections(kw.get("sampling_interval_ms") or 100),
    )


@pytest.fixture
def fake_model(monkeypatch):
    """모델 경계만 가로챈다. 좌표 계산과 렌더링은 진짜가 돈다."""
    _patch_model(monkeypatch)


# ---------- 실행 헬퍼 ----------


def _run(tmp_path, **cfg):
    """워커를 한 번 돌리고 (반환값, 산출물 dir, 진행 이벤트) 를 준다."""
    work = tmp_path / "work"
    src = _make_video(work / "source.mp4")
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "job1").mkdir(parents=True)
    (jobs_dir / "job1" / "progress.jsonl").touch()

    full = {"source": str(src), "out": str(work / "out.mp4"),
            "specs": [("m1", "/fake/model.pt")], "device": "cpu"}
    full.update(cfg)

    result = annotate_worker.run_annotate("job1", full, str(jobs_dir))
    events = [
        json.loads(line)
        for line in (jobs_dir / "job1" / "progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return result, work, events


def _phases(events):
    return [e["phase"] for e in events]


def _frames_of(path):
    """산출물의 모든 프레임을 배열로 읽는다 — 그려졌는지 대조용."""
    cap = cv2.VideoCapture(str(path))
    try:
        out = []
        while True:
            ok, frame = cap.read()
            if not ok:
                return out
            out.append(frame)
    finally:
        cap.release()


# ---------- 추론 없는 컷 (crop_source) ----------


def test_center_cut_produces_a_narrow_clip_without_any_model(tmp_path):
    result, work, events = _run(tmp_path, crop_source="center")

    assert result == {"status": "done", "frames": FRAMES}
    meta = video.probe(work / "out.mp4")
    assert (meta.width, meta.height) == (CROP_W, H)
    assert meta.frame_count == FRAMES
    assert _phases(events)[0] == "start"
    assert _phases(events)[-1] == "done"


def test_json_cut_follows_uploaded_keyframes(tmp_path):
    crop_json = tmp_path / "crop.json"
    crop_json.write_text(json.dumps({
        "source": {"width": W, "height": H},
        "crop": {"width": CROP_W, "height": H},
        "keyframes": [
            {"videoOffsetMs": 0, "x": 0},
            {"videoOffsetMs": 1000, "x": W - CROP_W},
        ],
    }))

    result, work, _ = _run(
        tmp_path, crop_source="json", crop_json_path=str(crop_json)
    )

    assert result["status"] == "done"
    assert video.probe(work / "out.mp4").width == CROP_W


def test_json_cut_accepts_the_legacy_samples_schema(tmp_path):
    """구버전 crop.json(samples/target_center_x)도 계속 받아야 한다."""
    crop_json = tmp_path / "crop.json"
    crop_json.write_text(json.dumps({
        "samples": [
            {"video_offset_ms": 0, "target_center_x": 120.0, "target_type": "ball"},
            {"video_offset_ms": 1000, "target_center_x": 500.0, "target_type": "ball"},
        ]
    }))

    result, work, _ = _run(tmp_path, crop_source="json", crop_json_path=str(crop_json))

    assert result["status"] == "done"
    assert video.probe(work / "out.mp4").width == CROP_W


# ---------- 크롭 추적 (검출 → 좌표 → 산출) ----------


def test_json_only_writes_coordinates_and_skips_encoding(tmp_path, fake_model):
    result, work, events = _run(
        tmp_path, crop_tracking=True, crop_output="none", object_tracking=False
    )

    assert result == {"status": "done", "frames": 0, "json_only": True}
    assert not (work / "out.mp4").exists()  # 렌더를 건너뛴다
    assert not (work / "._raw.mp4").exists()  # 중간 파일도 남기지 않는다
    plan = json.loads((work / "crop.json").read_text())
    assert plan["keyframes"]
    assert "crop_analyze" in _phases(events)
    assert "encoding" not in _phases(events)


def test_crop_video_output_cuts_to_the_vertical_window(tmp_path, fake_model):
    result, work, _ = _run(
        tmp_path, crop_tracking=True, crop_output="video", object_tracking=True
    )

    assert result == {"status": "done", "frames": FRAMES}
    meta = video.probe(work / "out.mp4")
    # 컷 모드는 object_tracking 을 무시하고 깨끗한 세로 클립만 낸다
    assert (meta.width, meta.height) == (CROP_W, H)
    assert (work / "crop.json").exists()  # 좌표는 언제나 남긴다


def test_crop_label_overlay_keeps_the_source_size(tmp_path, fake_model):
    result, work, _ = _run(
        tmp_path, crop_tracking=True, crop_output="label", object_tracking=False
    )

    assert result == {"status": "done", "frames": FRAMES}
    meta = video.probe(work / "out.mp4")
    assert (meta.width, meta.height) == (W, H)
    assert meta.frame_count == FRAMES


def test_object_tracking_only_keeps_the_source_size(tmp_path, fake_model):
    result, work, _ = _run(
        tmp_path, object_tracking=True, crop_tracking=False, crop_output="label"
    )

    assert result == {"status": "done", "frames": FRAMES}
    meta = video.probe(work / "out.mp4")
    assert (meta.width, meta.height) == (W, H)
    assert not (work / "crop.json").exists()  # 크롭을 끄면 좌표도 없다


def test_both_overlays_compose_on_one_clip(tmp_path, fake_model):
    result, work, events = _run(
        tmp_path, object_tracking=True, crop_tracking=True, crop_output="label",
        show_target_highlight=True,
    )

    assert result == {"status": "done", "frames": FRAMES}
    assert video.probe(work / "out.mp4").width == W
    assert (work / "crop.json").exists()
    assert _phases(events).count("crop_analyze") == 1  # 검출 패스는 한 번뿐


# ---------- 진행률 계약 ----------


@pytest.mark.parametrize(
    "mode, expect_encoding",
    [
        pytest.param({"crop_source": "center"}, True, id="추론없는-컷"),
        pytest.param(
            {"crop_tracking": True, "crop_output": "video", "object_tracking": False},
            True, id="크롭-컷",
        ),
        pytest.param(
            {"crop_tracking": True, "crop_output": "label", "object_tracking": False},
            True, id="크롭-오버레이",
        ),
        pytest.param({"crop_tracking": False, "object_tracking": True}, True, id="객체-추적"),
        pytest.param(
            {"crop_tracking": True, "crop_output": "none", "object_tracking": False},
            False, id="JSON만",
        ),
    ],
)
def test_progress_starts_and_ends_on_every_path(tmp_path, fake_model, mode, expect_encoding):
    """SSE 를 보는 쪽은 종료 페이즈로 스트림을 닫는다 — 경로마다 반드시 남아야 한다."""
    _, _, events = _run(tmp_path, **mode)
    phases = _phases(events)

    assert phases[0] == "start"
    assert phases[-1] == "done"
    assert ("encoding" in phases) is expect_encoding
    done_events = [e for e in events if e["phase"] == "done"]
    assert done_events[-1]["total"] == FRAMES


# ---------- 그리기가 실제로 픽셀을 바꾸는가 ----------


def test_crop_box_toggle_changes_the_rendered_pixels(tmp_path, fake_model):
    """토글이 꺼지면 같은 클립이 나와야 하고, 켜지면 달라야 한다.

    산출물 크기만 보면 그리기를 통째로 빼먹어도 통과한다 — 픽셀로 확인한다.
    """
    common = dict(crop_tracking=True, crop_output="label", object_tracking=False)
    _, on_dir, _ = _run(tmp_path / "on", **common, draw_crop_box=True)
    _, off_dir, _ = _run(
        tmp_path / "off", **common, draw_crop_box=False, show_dead_zone=False,
        show_center_line=False,
    )

    on, off = _frames_of(on_dir / "out.mp4"), _frames_of(off_dir / "out.mp4")
    assert len(on) == len(off) == FRAMES
    assert any(not np.array_equal(a, b) for a, b in zip(on, off))


def test_object_boxes_change_the_rendered_pixels(tmp_path, monkeypatch):
    """검출이 있을 때와 없을 때의 렌더가 달라야 한다 (박스를 실제로 그린다)."""
    _patch_model(monkeypatch)
    _, with_dir, _ = _run(
        tmp_path / "with", object_tracking=True, crop_tracking=False
    )
    _patch_model(monkeypatch, _FakeYOLONoBoxes)
    _, without_dir, _ = _run(
        tmp_path / "without", object_tracking=True, crop_tracking=False
    )

    a, b = _frames_of(with_dir / "out.mp4"), _frames_of(without_dir / "out.mp4")
    assert len(a) == len(b) == FRAMES
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


# ---------- 취소 · 실패 · 정리 ----------


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param({"crop_source": "center"}, id="추론없는-컷"),
        pytest.param(
            {"crop_tracking": True, "crop_output": "video", "object_tracking": False},
            id="크롭-컷",
        ),
        pytest.param(
            {"crop_tracking": True, "crop_output": "label", "object_tracking": False},
            id="크롭-오버레이",
        ),
        pytest.param(
            {"crop_tracking": False, "object_tracking": True}, id="객체-추적"
        ),
    ],
)
def test_cancel_stops_every_render_path(tmp_path, fake_model, mode):
    """취소 센티널은 렌더 경로 **넷 모두**에서 통해야 한다."""
    work = tmp_path / "work"
    src = _make_video(work / "source.mp4")
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "job1").mkdir(parents=True)
    (jobs_dir / "job1" / "progress.jsonl").touch()
    (jobs_dir / "job1" / "CANCEL").touch()  # 시작 전에 이미 취소

    cfg = {"source": str(src), "out": str(work / "out.mp4"),
           "specs": [("m1", "/fake/model.pt")], "device": "cpu", **mode}
    result = annotate_worker.run_annotate("job1", cfg, str(jobs_dir))

    assert result == {"status": "cancelled"}
    assert not (work / "out.mp4").exists()
    assert not (work / "._raw.mp4").exists()  # 중간 파일 정리


def test_failure_is_recorded_in_progress_and_reraised(tmp_path):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "job1").mkdir(parents=True)
    (jobs_dir / "job1" / "progress.jsonl").touch()

    with pytest.raises(Exception):
        annotate_worker.run_annotate(
            "job1",
            {"source": str(tmp_path / "missing.mp4"), "out": str(tmp_path / "out.mp4"),
             "specs": [("m1", "/fake/model.pt")], "device": "cpu"},
            str(jobs_dir),
        )

    events = [
        json.loads(line)
        for line in (jobs_dir / "job1" / "progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert events[-1]["phase"] == "error"
    assert events[-1]["msg"]


def test_the_source_video_is_always_consumed(tmp_path):
    """업로드된 원본은 일회성이다 — 성공하든 실패하든 남기지 않는다."""
    _, work, _ = _run(tmp_path, crop_source="center")

    assert not (work / "source.mp4").exists()

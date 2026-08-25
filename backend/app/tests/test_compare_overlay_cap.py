"""오버레이 상한 — 채점은 전수, 저장은 틀린 것이 많은 순으로 상한까지.

10만 장이면 `result.json` 이 기가 단위가 되어 워커가 OOM 킬러에 맞는다. 자식 프로세스가
그렇게 죽으면 종료 이벤트가 없어 화면에 영원히 running 으로 남으므로, 상한은 표시 문제가
아니라 런이 끝나느냐 마느냐의 문제다.
"""

import json

import yaml
from PIL import Image

from app.core.config import settings
from app.tests.test_compare_worker import _entry, fake_ultralytics  # noqa: F401
from app.workers import compare_worker
from app.workers.compare_worker import run_compare
from lib.labels.io import write_label_file

# "m1" 은 어느 이미지에서든 [0.10,0.10,0.30,0.30] 에 ball 을 0.9 로 내놓는다.
# 정답을 그 자리에 두면 흠이 없고, 다른 자리에 두면 오검출 1 + 놓침 n 이 쌓인다.
_HIT = (0.10, 0.10, 0.30, 0.30)
_ELSEWHERE = [(0.60, 0.60, 0.80, 0.80), (0.05, 0.70, 0.15, 0.90), (0.70, 0.05, 0.90, 0.25)]


def _seed_by_error_count(tmp_path, misses_per_image: list[int]):
    """이미지 i 의 정답을 `misses_per_image[i]` 개만큼 **빗나간 자리**에 둔다.

    빗나간 정답이 m 개면 그 이미지에서 틀린 수는 놓침 m + 오검출 1 이다(정답이 0 개면 0).
    """
    ds = tmp_path / "ds_cap"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    for i, misses in enumerate(misses_per_image):
        Image.new("RGB", (200, 100)).save(ds / "images" / f"img{i}.jpg")
        boxes = [(0, _HIT)] if misses == 0 else [(0, _ELSEWHERE[k]) for k in range(misses)]
        write_label_file(ds / "labels" / f"img{i}.txt", boxes)
    (ds / "data.yaml").write_text(yaml.safe_dump({"names": {0: "ball"}, "nc": 1}))
    return ds


def _run(job, ds, out_dir):
    (settings.jobs_dir / job).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_compare(
        job,
        {
            "project_id": "p1",
            "entries": [_entry("m1", "m1", "Model A", "m1.pt")],
            "dataset_dir": str(ds),
            "out_dir": str(out_dir),
            "conf": 0.4,
            "iou": 0.5,
            "device": "cpu",
        },
        str(settings.jobs_dir),
    )
    return json.loads((out_dir / "result.json").read_text())


def test_overlay_keeps_the_most_wrong_images_and_counts_every_scored_one(
    tmp_path, fake_ultralytics, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(compare_worker, "OVERLAY_LIMIT", 2)
    # img0 흠 없음(0) · img1 놓침1+오검출1(2) · img2 놓침3+오검출1(4)
    ds = _seed_by_error_count(tmp_path, [0, 1, 3])
    out_dir = tmp_path / "out_cap"

    result = _run("job_cap", ds, out_dir)

    # 채점은 전수로 했다 — 상한에 걸린 것은 저장뿐이다
    assert result["image_count"] == 3
    assert result["overlay_selection"] == {"criterion": "most_errors", "limit": 2, "kept": 2}
    assert len(result["images"]) == 2
    # 제일 덜 틀린 img0 이 밀려나고, 원래 순번 그대로 늘어선다
    assert [img["stem"] for img in result["images"]] == ["img1", "img2"]
    # 전수 채점이라 대표 숫자는 밀려난 이미지까지 반영한다: 정답 5개 중 1개만 맞혔다
    overall = result["per_entry"][0]["overall"]
    assert overall["tp"] == 1 and overall["fn"] == 4 and overall["fp"] == 2

    # 색인은 **원래 순번**이라야 오버레이 URL 이 어긋나지 않는다
    manifest = json.loads((out_dir / "images_manifest.json").read_text())
    assert set(manifest) == {"1", "2"}
    assert result["images"][0]["url"].split("?")[0].endswith("/images/1")


def test_overlay_selection_says_nothing_was_dropped_when_under_the_limit(
    tmp_path, fake_ultralytics, monkeypatch  # noqa: F811
):
    """상한에 걸리지 않은 런에서는 화면이 "표본입니다"라고 말하면 안 된다 —
    그래서 `kept` 가 `image_count` 와 같다는 것이 판별 기준이 된다."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    ds = _seed_by_error_count(tmp_path, [0, 2])
    out_dir = tmp_path / "out_uncapped"

    result = _run("job_uncapped", ds, out_dir)

    assert result["image_count"] == 2
    assert result["overlay_selection"]["kept"] == 2
    assert result["overlay_selection"]["limit"] == compare_worker.OVERLAY_LIMIT
    assert len(result["images"]) == 2

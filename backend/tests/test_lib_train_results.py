"""lib/train/results — 학습 산출물 읽기 계약.

ultralytics 가 쓴 파일이 차트 숫자의 유일한 출처다. 버전에 따라 산출물이 한 단계
아래 놓이고, 학습 중에 읽으면 마지막 줄이 잘려 있다 — 그 두 가지를 견디는지 본다.
"""

import json

from lib.train import results


def test_find_prefers_the_run_directory_itself(tmp_path):
    (tmp_path / "results.csv").write_text("epoch\n1\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "results.csv").write_text("epoch\n9\n")

    assert results.find(tmp_path, "results.csv") == tmp_path / "results.csv"


def test_find_falls_back_one_level_down(tmp_path):
    """일부 ultralytics 판은 산출물을 <run_dir>/<name>/ 아래에 쓴다."""
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "args.yaml").write_text("epochs: 10\n")

    assert results.find(tmp_path, "args.yaml") == tmp_path / "run1" / "args.yaml"


def test_find_returns_none_when_absent(tmp_path):
    assert results.find(tmp_path, "results.csv") is None


def test_read_results_csv_parses_numbers_and_strips_header_padding(tmp_path):
    """ultralytics 판에 따라 헤더 칸 앞에 공백이 붙는다."""
    path = tmp_path / "results.csv"
    path.write_text("epoch,   train/box_loss,  metrics/mAP50(B)\n1,1.5,0.42\n2,1.1,0.55\n")

    rows = results.read_results_csv(path)

    assert len(rows) == 2
    assert rows[0] == {"epoch": 1.0, "train/box_loss": 1.5, "metrics/mAP50(B)": 0.42}
    assert rows[1]["metrics/mAP50(B)"] == 0.55


def test_read_results_csv_keeps_non_numeric_values_as_text(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("epoch,note\n1,warmup\n")

    assert results.read_results_csv(path)[0]["note"] == "warmup"


def test_read_json_returns_the_default_when_missing_or_broken(tmp_path):
    assert results.read_json(tmp_path / "nope.json", default=[]) == []

    broken = tmp_path / "per_class.json"
    broken.write_text("{not json")
    assert results.read_json(broken, default=[]) == []


def test_read_json_returns_the_parsed_content(tmp_path):
    path = tmp_path / "per_class.json"
    path.write_text(json.dumps([{"cls": 0, "name": "ball"}]))

    assert results.read_json(path, default=[]) == [{"cls": 0, "name": "ball"}]


def test_read_jsonl_skips_blank_and_truncated_lines(tmp_path):
    """학습이 도는 중에 읽으면 마지막 줄이 반쯤 쓰여 있다 — 그 줄만 버린다."""
    path = tmp_path / "per_class_history.jsonl"
    path.write_text('{"epoch": 1}\n\n{"epoch": 2}\n{"epoch": 3, "met')

    assert results.read_jsonl(path) == [{"epoch": 1}, {"epoch": 2}]


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert results.read_jsonl(tmp_path / "nope.jsonl") == []

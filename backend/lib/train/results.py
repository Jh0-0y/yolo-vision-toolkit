"""학습이 남긴 산출물 읽기 — results.csv · per-class 지표.

**숫자의 출처는 ultralytics 가 쓴 파일이다.** 진행률 스트림(`progress.jsonl`)은
"지금 몇 에폭·어느 단계"만 알리고, 차트가 쓰는 값은 전부 여기서 읽는다. 같은
숫자를 두 경로로 나르지 않기 위해서다.

ultralytics 버전에 따라 산출물이 `<run_dir>/` 바로 아래일 수도, 한 단계
아래(`<run_dir>/<name>/`)일 수도 있어 `find` 가 양쪽을 본다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def find(run_dir: Path, name: str) -> Path | None:
    """산출물 하나를 찾는다. 루트 우선, 없으면 한 단계 아래. 없으면 None."""
    direct = run_dir / name
    if direct.exists():
        return direct
    found = sorted(run_dir.glob(f"*/{name}"))
    return found[0] if found else None


def read_results_csv(path: Path) -> list[dict]:
    """에폭당 한 줄. 숫자로 읽히는 칸은 float 으로, 아니면 원문 그대로 둔다.

    헤더에 붙은 공백은 떼고, 이름 없는 칸은 버린다 — ultralytics 가 쓰는
    results.csv 는 칸 앞에 공백이 붙어 있는 판이 있다.
    """
    rows: list[dict] = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for k, v in raw.items():
                key = (k or "").strip()
                if not key:
                    continue
                try:
                    row[key] = float(v)
                except (TypeError, ValueError):
                    row[key] = v
            rows.append(row)
    return rows


def read_json(path: Path, default):
    """깨졌거나 없으면 `default`. 지표 파일 하나 때문에 화면이 죽지 않게."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    """줄 단위 JSON. 깨진 줄은 건너뛴다 — 학습 중에 읽으면 마지막 줄이 잘려 있다."""
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows

"""업로드된 YOLO 데이터셋 zip 을 학습 가능한 형태로 푼다.

업로드되는 zip 은 출처가 제각각이다 — 다른 기계의 절대경로, `path:` 루트,
`data.yml`(y 하나), data.yaml 이 한 단계 아래 있는 구조. 학습 러너는 그중
**하나의 형태만** 받으므로(`<dir>/data.yaml`, train/val 은 그 yaml 기준 상대경로)
여기서 실제 디스크에 있는 것을 보고 다시 맞춘다.

HTTP 를 모른다. 형식이 틀리면 `DatasetError` 를 던지고, 그것을 몇 번 응답으로
옮길지는 호출자가 정한다.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from lib.formats import IMAGE_EXTS
from lib.labels.io import atomic_write_text


class DatasetError(Exception):
    """zip 이나 data.yaml 이 학습에 쓸 수 없는 형태다. 메시지는 사용자에게 보인다."""


def count_images(path: Path) -> int:
    """디렉터리면 안의 이미지 수, txt 목록 파일이면 비어 있지 않은 줄 수."""
    if not path.exists():
        return 0
    if path.is_file():  # train 이 txt 목록 파일을 가리킬 수 있다
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def normalize_data_yaml(yaml_path: Path) -> dict:
    """train/val 을 yaml 기준 상대경로로 다시 쓰고 `path` 키를 지운다.

    돌려주는 값은 `{"train": 장수, "val": 장수, "classes": 개수}`.
    train 경로를 어떤 후보로도 못 찾으면 `DatasetError`.
    """
    root = yaml_path.parent
    data = yaml.safe_load(yaml_path.read_text()) or {}
    base = data.get("path")

    def _resolve(key: str) -> str | None:
        val = data.get(key)
        if val is None:
            return None
        p = Path(str(val))
        candidates: list[Path] = []
        if not p.is_absolute():
            candidates.append(root / p)
            if base and not Path(str(base)).is_absolute():
                candidates.append(root / str(base) / p)
        # 절대경로이거나 위에서 못 찾은 경우: 꼬리 경로를 압축 해제 루트에 맞춰 본다
        candidates.append(root / p.name)
        if len(p.parts) >= 2:
            candidates.append(root / Path(*p.parts[-2:]))
        for c in candidates:
            if c.exists():
                return c.relative_to(root).as_posix()
        return None

    train = _resolve("train")
    if train is None:
        raise DatasetError("Cannot resolve the train path in data.yaml")
    val = _resolve("val") or train

    data.pop("path", None)
    data["train"] = train
    data["val"] = val
    atomic_write_text(yaml_path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    names = data.get("names") or {}
    return {
        "train": count_images(root / train),
        "val": count_images(root / val),
        "classes": int(data.get("nc") or len(names)),
    }


def find_data_yaml(dest: Path) -> Path | None:
    """루트, 없으면 한 단계 아래에서 data.yaml(또는 .yml)을 찾는다. 없으면 None.

    zip 이 폴더 하나를 감싸고 있는 형태가 흔해서 두 단계를 본다. 압축 해제된
    YOLO 데이터셋이면 무엇이든 쓰므로 학습 전용은 아니다 — 모델 비교의 테스트셋도
    이 함수를 쓴다."""
    for candidate in ("data.yaml", "data.yml"):
        if (dest / candidate).exists():
            return dest / candidate
    found = sorted(list(dest.glob("*/data.yaml")) + list(dest.glob("*/data.yml")))
    return found[0] if found else None


def extract_zip(
    zip_path: Path,
    dest: Path,
    *,
    dataset_id: str,
    name: str,
    auto_delete: bool,
    now: datetime,
) -> dict:
    """zip 을 `dest` 에 풀고 data.yaml 을 정규화한 뒤 dataset.json 을 쓴다.

    실패하면 `dest` 를 통째로 지우고 다시 raise 한다 — 반쯤 풀린 데이터셋이
    목록에 뜨면 안 된다.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile as e:
            raise DatasetError("Corrupted zip file") from e

        yaml_path = find_data_yaml(dest)
        if yaml_path is None:
            raise DatasetError(
                "data.yaml not found in the zip (a YOLO-format dataset is required)"
            )
        if yaml_path.name != "data.yaml":  # 학습 러너는 이 이름만 찾는다
            renamed = yaml_path.with_name("data.yaml")
            yaml_path.rename(renamed)
            yaml_path = renamed

        counts = normalize_data_yaml(yaml_path)
        if counts["train"] == 0:
            raise DatasetError("No train images found")

        meta = {
            "id": dataset_id,
            "name": name,
            "yaml_dir": yaml_path.parent.relative_to(dest).as_posix(),
            "created_at": now.isoformat(),
            # True 면 이 데이터셋을 쓴 학습이 성공했을 때 삭제된다
            # (train_manager._watch) — 일회성 외부 zip 용
            "auto_delete": auto_delete,
            **counts,
        }
        atomic_write_text(dest / "dataset.json", json.dumps(meta, ensure_ascii=False))
        return meta
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise

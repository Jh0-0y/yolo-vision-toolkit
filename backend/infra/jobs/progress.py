"""잡 진행률 파일(`progress.jsonl`) 읽고 쓰기.

자식 프로세스는 DB 를 쓸 수 없어 진행 상황을 이 파일로만 남긴다. API 프로세스가
이 파일을 읽어 SSE 로 흘려보내고, **끝난 뒤의 사실만** DB 에 옮긴다.

한 줄이 이벤트 하나(JSON)다. 줄 단위라 쓰는 쪽과 읽는 쪽이 락 없이 만나고,
읽는 쪽은 바이트 오프셋만 기억하면 이어 읽을 수 있다.

**무엇을 담을지는 각 기능이 정한다.** 여기서 강제하는 건 `ts` 하나뿐이다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

FILENAME = "progress.jsonl"


def path_in(job_dir: Path) -> Path:
    return job_dir / FILENAME


def emit(progress_path: Path, event: dict) -> None:
    """이벤트 한 줄을 덧붙인다. `ts` 는 여기서 붙이므로 호출자가 넣지 않는다."""
    with open(progress_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}, ensure_ascii=False) + "\n")


def read(progress_path: Path, offset: int = 0) -> tuple[list[dict], int]:
    """바이트 오프셋부터 읽는다. `(이벤트 목록, 다음 오프셋)` 을 돌려준다.

    쓰는 쪽이 아직 줄을 다 못 썼을 수 있으므로 **개행으로 끝나는 데까지만** 읽고
    잘린 마지막 줄은 다음 호출로 넘긴다.
    """
    if not progress_path.exists():
        return [], offset
    with open(progress_path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    consumed = chunk.rfind(b"\n") + 1
    events = []
    for line in chunk[:consumed].splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events, offset + consumed

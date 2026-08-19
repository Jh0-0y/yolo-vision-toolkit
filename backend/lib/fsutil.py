"""파일을 자리에 들이는 방법 — 로직 없는 공용 어휘.

`lib/formats.py` 처럼 어디서든 읽는 최상위 모듈이다. 크롭 런의 가로 영상과
데이터셋 내보내기가 **같은 문제**를 갖는다: 원본을 그대로 한 벌 더 두어야 하는데
통째로 복사하면 용량이 배로 든다.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> str:
    """`src` 를 `dst` 자리에 들인다. 하드링크가 되면 링크, 안 되면 복사.

    돌려주는 값은 실제로 한 일(`"link"` · `"copy"`)이다. 링크든 복사든 **원본을
    지워도 `dst` 는 온전하다** — 그래서 산출물이 원본의 수명에 매이지 않는다.

    하드링크는 같은 볼륨 안에서만 된다. 다른 볼륨이면 `OSError` 가 나고, 그때는
    용량을 쓰더라도 복사한다.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"

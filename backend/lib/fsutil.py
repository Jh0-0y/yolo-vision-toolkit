"""파일을 자리에 들이는 방법과 이름 짓는 방법 — 로직 없는 공용 어휘.

`lib/formats.py` 처럼 어디서든 읽는 최상위 모듈이다. 크롭 런의 가로 영상과
데이터셋 내보내기가 **같은 문제**를 갖는다: 원본을 그대로 한 벌 더 두어야 하는데
통째로 복사하면 용량이 배로 든다.

이름도 마찬가지로 한 군데서 정한다 — 영상에서 뽑은 프레임과 zip 으로 들여온
이미지가 **같은 규칙**을 따라야 한쪽만 깨지는 일이 없다.
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path

# 파일명에서 **반드시** 걷어내야 하는 것들. 한글·공백·괄호는 건드리지 않는다 —
# 이름을 알아볼 수 없게 만드는 것이 목적이 아니라, 깨지는 것만 막는 것이 목적이다.
#
#   / \      경로 구분자
#   # ? %    URL 을 깬다 (# 은 프래그먼트, ? 는 쿼리, % 는 잘못된 이스케이프)
#   : * " < > |   윈도우에서 불법 — 내보낸 zip 을 윈도우에서 풀 때 그 파일만 실패한다
#   제어문자
_UNSAFE = re.compile(r'[/\\#?%:*"<>|\x00-\x1f\x7f]+')

# 파일명 한계는 보통 255**바이트**다. 타일 접미사(`_r1c3`)와 번호(`_00001`),
# 확장자까지 뒤에 붙으므로 넉넉히 남겨 둔다.
_MAX_STEM_BYTES = 150


def safe_stem(name: str, fallback: str = "file") -> str:
    """파일명(또는 경로)에서 **안전한 stem** 을 만든다.

    지금까지는 영숫자 외를 전부 `_` 로 밀어서 한글 이름이 통째로 사라졌다. 그러면
    한글 이름 영상 두 개가 같은 stem 을 갖고 서로 덮어쓴다. 그래서 **깨지는 문자만**
    바꾸고 나머지는 그대로 둔다.

    - 앞의 `.` 은 없앤다. 숨김 파일이 되면 목록에서 빠지고 서빙도 막힌다.
    - macOS 가 NFD 로 돌려주는 자리가 있어 **NFC 로 맞춰** 둔다. 그래야 검수·분할
      기록의 키와 디스크의 이름이 어긋나지 않는다.
    - 남는 것이 없으면 `fallback`.
    """
    stem = Path(name).stem
    stem = unicodedata.normalize("NFC", stem)
    stem = _UNSAFE.sub("_", stem)
    stem = stem.strip().strip(".").strip()
    while len(stem.encode()) > _MAX_STEM_BYTES:
        stem = stem[:-1]
    return stem or fallback


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

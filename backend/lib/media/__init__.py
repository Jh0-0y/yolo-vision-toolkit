"""미디어 파일 다루기 — 프레임 추출 · 타일링 · 썸네일.

    extract     영상 -> 학습용 이미지 (샘플링 · 중복 제거 · 타일 저장)
    tiling      큰 이미지를 겹치는 타일 격자로 쪼개기
    thumbnails  256px 캐시 썸네일

전부 픽셀만 만진다 — 잡도 DB도 모른다. `extract` 는 오래 걸리므로 진행률을
`emit` 콜백으로, 취소를 `cancel_path` 로 **인자로 받는다.**

확장자 어휘는 여기가 아니라 `lib/formats.py` 에 있다 — 이미지 목록을 훑는 코드가
이 패키지 밖에도 많기 때문이다.
"""

from lib.media import extract, thumbnails, tiling

__all__ = ["extract", "thumbnails", "tiling"]

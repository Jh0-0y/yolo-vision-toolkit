"""파일 확장자 어휘 — "이건 이미지인가 영상인가".

로직이 아니라 **공용 어휘**다. 주제 패키지끼리는 서로의 로직을 import 하지
않는다는 규칙의 예외로, 여기(와 앞으로 생길 `lib/types.py`)는 어디서든 읽어도
된다. 어휘가 한 곳에 있어야 "왜 이 화면만 .tiff 를 못 받지" 같은 어긋남이
생기지 않는다.

전부 소문자 · 점 포함이다. 비교할 때 `path.suffix.lower()` 를 쓴다.
"""

from __future__ import annotations

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

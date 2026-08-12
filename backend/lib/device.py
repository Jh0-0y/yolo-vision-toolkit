"""디바이스 문자열 해석 — "auto" 를 실제로 쓸 장치로 바꾼다.

설정을 모른다. `"auto"` 를 무엇으로 볼지만 안다. 기본값을 채우는 일(요청에
디바이스가 없을 때 무엇을 쓸지)은 앱의 몫이라 `app.core.config.resolve_device`
가 맡는다.

torch 는 함수 안에서 import 한다 — 이 모듈을 읽는 것만으로 API 프로세스에
프레임워크가 올라오면 안 된다.
"""

from __future__ import annotations


def resolve(requested: str) -> str:
    """`"auto"` 면 쓸 수 있는 가속기를 고르고, 아니면 받은 값을 그대로 돌려준다.

    반환값은 ultralytics 가 받는 형식이다 — `"0"`(첫 CUDA 장치) · `"mps"` · `"cpu"`.
    """
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

"""잡 실행의 배관 — 진행률 · 취소 · 잡 디렉터리.

기능(무엇을 계산하는가)이 아니라 시스템 관심사(어떻게 알리고 어떻게 멈추는가)만
담는다. 이벤트에 **무엇을** 담을지는 각 기능이 정하고, 여기서는 **형식**만 정한다.

순수 계산(`lib/`)은 이 모듈을 import 하지 않는다. 진행률이 필요하면 콜백으로
받는다 — 그래야 웹 없이도 그 계산을 쓸 수 있다.
"""

from infra.jobs.dir import JobDir, at
from infra.jobs.progress import emit, read

__all__ = ["JobDir", "at", "emit", "read"]

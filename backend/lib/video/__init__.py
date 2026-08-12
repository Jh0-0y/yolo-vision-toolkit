"""영상 입출력 — 규격 읽기와 인코딩.

무거운 의존(cv2)은 **함수 안에서** import 한다. API 프로세스가 이 모듈을 읽어도
프레임워크가 딸려 오지 않게 하기 위해서다.
"""

from lib.video.encode import require_ffmpeg, to_h264
from lib.video.probe import VideoMeta, probe

__all__ = ["VideoMeta", "probe", "require_ffmpeg", "to_h264"]

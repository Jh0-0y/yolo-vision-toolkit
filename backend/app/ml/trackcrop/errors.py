"""파이프라인 오류.

원본(LivePick)의 ERR 계약(Fail API용 error_type/code)을 단순화했다.
호출자가 실패 원인을 구분할 수 있도록 code 문자열만 남긴다.
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    VIDEO_DECODE_FAILED = "VIDEO_DECODE_FAILED"
    OBJECT_DETECTION_FAILED = "OBJECT_DETECTION_FAILED"


class TrackCropError(Exception):
    """추적·크롭 좌표 계산 중 발생한 오류."""

    def __init__(self, code: ErrorCode, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

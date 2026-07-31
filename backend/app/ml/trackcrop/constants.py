"""추적·크롭 좌표 계산 정책값 (고정값).

LivePick CropWorker의 MVP 고정값에서 파이프라인 계산에 필요한 것만 옮겨온 것이다.
Job/API/S3/스키마 계약값은 제외했다 — 이 모듈은 순수 계산만 한다.

값 변경 시 결과(Crop 좌표)가 달라지므로, 다른 규격을 쓰려면 여기만 수정한다.
"""

# 입력 영상 규격 (가로형)
SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080

# 출력(세로형) 규격 — 참고용
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# Source Crop 영역 — width = round_even(1080 * 1080 / 1920 = 607.5) = 608
CROP_WIDTH = 608
CROP_HEIGHT = 1080
CROP_Y = 0

# Crop X 허용 범위: 0 .. (SOURCE_WIDTH - CROP_WIDTH)
CROP_X_MIN = 0
CROP_X_MAX = SOURCE_WIDTH - CROP_WIDTH  # 1312

# 분석 정책
SAMPLING_INTERVAL_MS = 100
MAX_MOVE_PX_PER_SECOND = 1200
BALL_LOST_HOLD_MS = 1000
PLAYER_LOST_HOLD_MS = 1500

# 화면 중앙 Fallback (장시간 미검출 시 target_center_x)
CENTER_FALLBACK_X = SOURCE_WIDTH // 2  # 960

# 크롭 데드존 — 공이 크롭 중심 ±(폭/2) 안이면 크롭 고정 (작은 흔들림 무시)
DEAD_ZONE_WIDTH = 208

# Target 가중 중심 (공 + 주요 선수 군집)
BALL_WEIGHT = 0.7
PLAYER_GROUP_WEIGHT = 0.3

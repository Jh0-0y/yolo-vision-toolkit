// 연구실 크롭 런 — 시작 · 목록 · 상세 · 진행률 · 산출물 · 복제.
//
// 옛 크롭 랩은 시작(predict 계열)과 읽기(crops)가 갈라져 있었다. 여기서는 한 곳이다
// — 시작하면 런 하나를 그대로 돌려받으므로 화면은 바로 상세로 넘어간다.
import { BASE, api } from './http'


/** adaptive-crop 런타임 튜닝 오버라이드 — 비운 값은 보내지 않아 라이브러리 기본값 사용.
 *  키 이름은 백엔드 `ClipPlanConfig` 필드명과 1:1 이다. */
export interface TrackcropOverrides {
  // 검출/지오메트리 정책
  sampling_interval_ms?: number
  dead_zone_width?: number
  max_move_px_per_second?: number
  ball_weight?: number
  // tracklet 쪼개기 / 스티칭
  ball_max_speed_px_s?: number
  split_base_px?: number
  stitch_max_gap_ms?: number
  stitch_base_px?: number
  stitch_velocity_cap_ms?: number
  // 트랙 채점
  w_travel?: number
  w_interaction?: number
  w_span?: number
  travel_norm_px?: number
  min_track_score?: number
  absorb_allow_scale?: number
  possession_margin?: number
  // 타깃 결정
  prune_dev_px?: number
  interp_max_gap_ms?: number
  use_carrier?: boolean
  // 경로 최적화
  w_follow?: number
  w_inside?: number
  w_vel?: number
  w_acc?: number
  irls_iters?: number
  min_follow_conf?: number
}

export interface CropDetector {
  model_id: string
  mode: 'full' | 'tiled'
  conf?: number | null
  imgsz?: number
  tile_size?: number
  stride?: number
  merge_iou?: number
}

/** 그리기 도구 — 개별 토글. **하나도 켜지 않으면** wide.mp4 는 원본 사본이다. */
export interface CropToggles {
  obj_boxes: boolean // 추론 박스
  player_trails: boolean // 선수 궤적
  ball_trail: boolean // 공 궤적
  target_highlight: boolean // 타깃
  crop_box: boolean // 크롭 박스
  dead_zone: boolean // 데드존
  center_line: boolean // 중앙선
}

export const DEFAULT_TOGGLES: CropToggles = {
  obj_boxes: false,
  player_trails: false,
  ball_trail: false,
  target_highlight: false,
  crop_box: true,
  dead_zone: false,
  center_line: true,
}

export interface LabCropCreate {
  name: string
  source_video_id: string
  detectors: CropDetector[]
  overrides: TrackcropOverrides
  crop_w: number
  crop_h: number
  toggles: CropToggles
}

export interface LabCropRunOut {
  id: string
  name: string
  created_at: string
  source_video_id: string
  source_name: string
  models: { model_id: string; name: string; mode: string; conf: number | null }[]
  overrides: TrackcropOverrides
  crop_w: number
  crop_h: number
  // 소스에 맞춰 실제로 쓰인 창 — 요청값이 소스보다 크면 clamp 된 값이 온다
  applied_w: number
  applied_h: number
  toggles: Partial<CropToggles>
  wide_kind: 'link' | 'copy' | 'render' | ''
  status: 'running' | 'done' | 'error' | 'cancelled'
  error: string | null
  summary: {
    keyframeCount?: number
    ballTrackingCoverage?: number
    playerTrackingCoverage?: number
    fallbackCoverage?: number
  } | null
  has_crop_json: boolean
  has_wide: boolean
  has_cut: boolean
  size_bytes: number
}

export interface LabCropProgressEvent {
  phase: 'start' | 'detect' | 'crop_analyze' | 'render' | 'encoding' | 'done' | 'error' | 'cancelled'
  done?: number
  total?: number
  msg?: string
  wide_kind?: string
}

export const createLabCropRun = (body: LabCropCreate) =>
  api.post<LabCropRunOut>(`/lab/crops`, body)

export const listLabCropRuns = () => api.get<LabCropRunOut[]>(`/lab/crops`)

export const getLabCropRun = (cropId: string) =>
  api.get<LabCropRunOut>(`/lab/crops/${cropId}`)

export const cancelLabCropRun = (cropId: string) =>
  api.post(`/lab/crops/${cropId}/cancel`)

/** "이 설정 그대로 다시, 모델만 바꿔서" — 모델 업그레이드 전후를 즉시 비교한다. */
export const duplicateLabCropRun = (
  cropId: string,
  body: { name?: string; detectors?: CropDetector[] } = {},
) => api.post<LabCropRunOut>(`/lab/crops/${cropId}/duplicate`, body)

export const renameLabCropRun = (cropId: string, name: string) =>
  api.patch<LabCropRunOut>(`/lab/crops/${cropId}`, { name })

export const deleteLabCropRun = (cropId: string) =>
  api.delete<void>(`/lab/crops/${cropId}`)

export const labCropJsonUrl = (cropId: string) =>
  `${BASE}/lab/crops/${cropId}/crop.json`

export const labWideVideoUrl = (cropId: string) =>
  `${BASE}/lab/crops/${cropId}/wide.mp4`

export const labCutVideoUrl = (cropId: string) =>
  `${BASE}/lab/crops/${cropId}/crop.mp4`

export function subscribeLabCropEvents(
  cropId: string,
  onEvent: (ev: LabCropProgressEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}/lab/crops/${cropId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as LabCropProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') source.close()
  })
  source.onerror = () => {
    // EventSource 는 일시적 끊김을 스스로 재시도한다 — CLOSED 만 진짜 실패다
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}

/** 타깃 타입의 색 — 정의는 백엔드 `lib/crop/palette.py` 하나뿐이다. */
export interface CropPalette {
  target_colors: Record<string, string>
  target_line_color: string
  dead_zone_color: string
  min_confidence_alpha: number
}

export const getCropPalette = () => api.get<CropPalette>('/system/crop-palette')

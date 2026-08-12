// 영상 오버레이 렌더와 크롭 컷 — **시작**만 여기 있다.
//
// 시작하고 나면 그 잡은 크롭 런이 된다. 진행률·결과·삭제는 전부 `crops.ts` 를
// 쓴다 (`job_id` 가 곧 crop run id 다).

import { api } from '../http'
import type { PredictParams } from './predict'


export interface TestJobStart {
  job_id: string
}

// "none" = JSON만 | "label" = 오버레이 그리기 | "video" = 세로 크롭 컷
export type CropOutput = 'none' | 'label' | 'video'

/** adaptive-crop 런타임 튜닝 오버라이드 — 비운 값은 보내지 않아 라이브러리 기본값 사용. */
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

// 검출기 엔트리 — 모두 대등. 첫 Full scan 엔트리가 ByteTrack 추적을 맡고
// 나머지는 공 검출에 기여한다 (백엔드에서 자동 유도).
export interface DetectorPayload {
  model_id: string
  mode: 'full' | 'tiled'
  conf?: number
  imgsz?: number
  tile_size?: number
  stride?: number
  merge_iou?: number
}

export interface BallDetectorOpts {
  detectors?: DetectorPayload[]
}

// 검출기 엔트리를 폼에 싣는다 — annotate 와 live 가 같은 형식을 보낸다.
export function appendBall(form: FormData, opts: BallDetectorOpts) {
  if (opts.detectors?.length) form.append('detectors', JSON.stringify(opts.detectors))
}

export function startAnnotate(opts: {
  file: File
  modelIds: string[]
  projectId: string
  params: PredictParams
  objectTracking: boolean
  cropTracking: boolean
  cropOutput: CropOutput
  drawCropBox?: boolean
  showDeadZone?: boolean
  showCenterLine?: boolean
  showTargetHighlight?: boolean
  overrides?: TrackcropOverrides
} & BallDetectorOpts): Promise<TestJobStart> {
  const form = new FormData()
  form.append('file', opts.file)
  form.append('model_ids', opts.modelIds.join(','))
  form.append('project_id', opts.projectId)
  if (opts.params.conf != null) form.append('conf', String(opts.params.conf))
  form.append('iou_wbf', String(opts.params.iou_wbf))
  // imgsz는 백엔드에서 1920 고정 — 호환 위해 값은 보내되 무시됨
  form.append('imgsz', String(opts.params.imgsz))
  if (opts.params.device) form.append('device', opts.params.device)
  form.append('object_tracking', String(opts.objectTracking))
  form.append('crop_tracking', String(opts.cropTracking))
  form.append('crop_output', opts.cropOutput)
  if (opts.drawCropBox != null) form.append('draw_crop_box', String(opts.drawCropBox))
  if (opts.showDeadZone != null) form.append('show_dead_zone', String(opts.showDeadZone))
  if (opts.showCenterLine != null) form.append('show_center_line', String(opts.showCenterLine))
  if (opts.showTargetHighlight != null)
    form.append('show_target_highlight', String(opts.showTargetHighlight))
  if (opts.overrides) form.append('overrides', JSON.stringify(opts.overrides))
  appendBall(form, opts)
  return api.upload<TestJobStart>('/predict/annotate', form)
}

// Crop-cut: make a vertical crop clip with NO inference.
//   mode="json"  → follow an uploaded crop.json / mode="center" → fixed centre.
export function startCropCut(opts: {
  file: File
  projectId: string
  mode: 'json' | 'center'
  cropJson?: File
}): Promise<TestJobStart> {
  const form = new FormData()
  form.append('file', opts.file)
  form.append('project_id', opts.projectId)
  form.append('mode', opts.mode)
  if (opts.cropJson) form.append('crop_json', opts.cropJson)
  return api.upload<TestJobStart>('/predict/crop-cut', form)
}

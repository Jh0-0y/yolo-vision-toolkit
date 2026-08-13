// 라이브 크롭 프리뷰 — 검출 1회, 튜닝마다 좌표만 다시 계산.

import { BASE, api } from '../http'
import { appendBall } from './annotate'
import type { BallDetectorOpts, TestJobStart, TrackcropOverrides } from './annotate'


// 검출 패스와 오버레이 렌더가 같은 스트림 형식을 쓴다 — 다른 것은 중간 phase 뿐이다.
export interface LiveProgress {
  phase: 'start' | 'detect' | 'render' | 'encoding' | 'done' | 'error' | 'cancelled'
  done?: number
  total?: number
  msg?: string
}

// One sampled frame's detections (ByteTrack boxes), in source-pixel coordinates.
export interface LiveDetection {
  object_type: 'ball' | 'player'
  track_id: number | null
  bbox_x: number
  bbox_y: number
  bbox_width: number
  bbox_height: number
  confidence: number
}

export interface LiveSample {
  video_offset_ms: number
  detections: LiveDetection[]
}

// Cached detection result + geometry meta (the crop trajectory comes from livePlan).
export interface LiveResult {
  detect_id: string
  source_width: number
  source_height: number
  fps: number
  duration_ms: number
  sampling_interval_ms: number
  sample_count: number
  detections: LiveSample[]
}

// One point of the target-centre trajectory (source-pixel X at a sample time).
export interface CropSample {
  video_offset_ms: number
  target_center_x: number
  target_type: 'ball' | 'ball_player' | 'player_group' | 'center'
  confidence: number
}

// Selected ball / carrier bbox at a sample time (target-highlight overlay). [x,y,w,h].
export interface CropDebug {
  video_offset_ms: number
  ball_bbox: [number, number, number, number] | null
  carrier_bbox: [number, number, number, number] | null
}

// Crop coordinates recomputed from cached detections for a given set of overrides.
// 계약 스키마(camelCase) + 라이브 오버레이용 내부 확장(samples/debug — 스펙 외).
// 파일로 내보낼 때는 samples/debug를 제거해 스펙만 남긴다.
export interface CropPlan {
  schemaVersion: number
  jobId: string
  gameId: string
  clipCandidateId: string
  sourceContentOutputId: string
  sourceMediaAssetId: string
  source: { width: number; height: number; durationMs: number }
  crop: { width: number; height: number; y: number; interpolation: string }
  keyframes: { videoOffsetMs: number; x: number; targetType: string; confidence: number }[]
  summary: Record<string, number>
  samples: CropSample[]
  debug: CropDebug[]
}

export function startLive(opts: {
  file: File
  modelIds: string[]
  projectId: string
  conf?: number
  imgsz: number
  device?: string | null
  samplingIntervalMs?: number
} & BallDetectorOpts): Promise<TestJobStart> {
  const form = new FormData()
  form.append('file', opts.file)
  form.append('model_ids', opts.modelIds.join(','))
  form.append('project_id', opts.projectId)
  if (opts.conf != null) form.append('conf', String(opts.conf))
  form.append('imgsz', String(opts.imgsz))
  if (opts.device) form.append('device', opts.device)
  if (opts.samplingIntervalMs != null)
    form.append('sampling_interval_ms', String(opts.samplingIntervalMs))
  appendBall(form, opts)
  return api.upload<TestJobStart>('/predict/live', form)
}

export function subscribeLiveEvents(
  jobId: string,
  onEvent: (ev: LiveProgress) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}/predict/live/${jobId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as LiveProgress
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') source.close()
  })
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}

export const cancelLiveJob = (jobId: string) => api.post(`/predict/live/${jobId}/cancel`)

export const liveVideoUrl = (detectId: string) => `${BASE}/predict/live/${detectId}/video`

// 세션이 아직 살아 있는지 — 기억해 둔 detect_id 로 돌아왔을 때 먼저 묻는다.
// 캐시는 TTL 로 사라지므로 서버만이 이걸 안다.
export interface LiveSessionStatus {
  status: 'running' | 'done' | 'error' | 'cancelled' | 'expired'
  msg: string | null
  has_render: boolean // 구워 둔 오버레이 영상이 아직 있나
}

export const getLiveStatus = (detectId: string) =>
  api.get<LiveSessionStatus>(`/predict/live/${detectId}/status`)

// 오버레이 렌더 — 캐시된 검출 + 현재 튜닝으로 그림 그려진 영상을 굽는다 (추론 없음)
export const startLiveRender = (
  detectId: string,
  overrides: TrackcropOverrides,
  toggles: Record<string, boolean>,
) => api.post<TestJobStart>(`/predict/live/${detectId}/render`, { overrides, toggles })

export const liveRenderVideoUrl = (detectId: string) =>
  `${BASE}/predict/live/${detectId}/render-video`

export const getLiveResult = (jobId: string) => api.get<LiveResult>(`/predict/live/${jobId}/result`)

// Cheap crop-coordinate recompute — call on every tuning change; no inference happens.
export const livePlan = (
  detectId: string,
  overrides: TrackcropOverrides,
  collectDebug: boolean,
) =>
  api.post<CropPlan>(`/predict/live/${detectId}/plan`, {
    overrides,
    collect_debug: collectDebug,
  })

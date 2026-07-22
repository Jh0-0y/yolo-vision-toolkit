const BASE = '/api'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    const body = await res.text()
    throw new ApiError(res.status, body || res.statusText)
  }
  return res.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
  upload: <T>(path: string, form: FormData) =>
    request<T>(path, { method: 'POST', body: form }),
}

export interface Health {
  status: string
  data_dir: string
}

export interface DeviceInfo {
  device: string
  requested: string
  accelerator: 'cuda' | 'mps' | 'cpu'
  accelerated: boolean
  device_name: string
  device_label: string
  cuda_available: boolean
  mps_available: boolean
  torch_version: string
  platform: string
  vram_total_mb: number | null
  vram_used_mb: number | null
}

export interface ModelOut {
  id: string
  name: string
  classes: Record<number, string>
  task: string
  created_at: string
  source: 'upload' | 'official'
}

export interface OfficialModel {
  name: string
  default: boolean
}

export interface ProjectOut {
  id: string
  name: string
  created_at: string
}

export interface StatsOut {
  raw: number
  confirmed: number
  review: number
  review_pending: number
  classes: { id: number; name: string; sources: string[] }[]
}

export interface ImageItem {
  name: string
  thumb: string
  url: string
}

export interface ImagePage {
  total: number
  page: number
  size: number
  items: ImageItem[]
}

export interface JobOut {
  id: string
  project_id: string
  status: string
  config: Record<string, unknown>
  result: {
    total: number
    confirmed: number
    review: number
    negative: number
  } | null
  error: string | null
  created_at: string
}

export interface JobProgressEvent {
  phase: 'inference' | 'done' | 'error' | 'cancelled'
  done?: number
  total?: number
  confirmed?: number
  review?: number
  negative?: number
  msg?: string
}

export interface ExportOut {
  id: string
  created_at: string
  val_split: number
  seed: number
  train: number
  val: number
  classes: number
  size_bytes: number
}

export interface ReviewQueueItem {
  id: string
  stem: string
  uncertainty: number
  n_flagged: number
  thumb: string | null
}

export interface ReviewQueue {
  total: number
  page: number
  size: number
  items: ReviewQueueItem[]
}

export interface ReviewBox {
  id: string | null
  cls: number
  xyxy_n: [number, number, number, number]
  score?: number | null
  status?: string | null
  reason?: string | null
  sources?: { model: string; score: number }[] | null
}

export interface ReviewItemDetail {
  id: string
  project_id: string
  stem: string
  status: string
  uncertainty: number
  image_url: string | null
  width: number
  height: number
  boxes: ReviewBox[]
  original_boxes: ReviewBox[]
  classes: { id: number; name: string; sources: string[] }[]
}

export interface TrainDataset {
  project_id: string
  project_name: string
  export_id: string
  train: number
  val: number
  classes: number
  created_at: string
}

export interface TrainRunOut {
  id: string
  name: string
  status: string
  dataset_path: string
  base_model_id: string
  base_model_name: string | null
  params: Record<string, number | string>
  metrics: Record<string, number> | null
  error: string | null
  created_at: string
  finished_at: string | null
}

export interface TrainEpochEvent {
  phase: 'epoch' | 'done' | 'error' | 'cancelled'
  epoch?: number
  epochs?: number
  metrics?: Record<string, number>
  msg?: string
}

export function subscribeTrainEvents(
  runId: string,
  onEvent: (ev: TrainEpochEvent) => void,
): () => void {
  const source = new EventSource(`${BASE}/train/runs/${runId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as TrainEpochEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') {
      source.close()
    }
  })
  return () => source.close()
}

export interface VideoUploadParams {
  target_fps: number
  max_frames: number
  start_sec: number
  end_sec: number | null
  dedup: boolean
  dedup_threshold: number
}

export interface VideoProgressEvent {
  phase: 'start' | 'extract' | 'done' | 'error' | 'cancelled'
  saved?: number
  scanned?: number
  total_frames?: number
  skipped_dup?: number
  src_fps?: number
  step?: number
  msg?: string
}

export function uploadVideo(projectId: string, file: File, params: VideoUploadParams) {
  const form = new FormData()
  form.append('file', file)
  form.append('target_fps', String(params.target_fps))
  form.append('max_frames', String(params.max_frames))
  form.append('start_sec', String(params.start_sec))
  if (params.end_sec != null) form.append('end_sec', String(params.end_sec))
  form.append('dedup', String(params.dedup))
  form.append('dedup_threshold', String(params.dedup_threshold))
  return api.upload<{ video_id: string; filename: string; status: string }>(
    `/projects/${projectId}/videos`,
    form,
  )
}

export function subscribeVideoEvents(
  projectId: string,
  videoId: string,
  onEvent: (ev: VideoProgressEvent) => void,
): () => void {
  const source = new EventSource(`${BASE}/projects/${projectId}/videos/${videoId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as VideoProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') {
      source.close()
    }
  })
  return () => source.close()
}

export function subscribeJobEvents(
  jobId: string,
  onEvent: (ev: JobProgressEvent) => void,
): () => void {
  const source = new EventSource(`${BASE}/jobs/${jobId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as JobProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') {
      source.close()
    }
  })
  source.onerror = () => {
    // EventSource auto-reconnects; nothing to do
  }
  return () => source.close()
}

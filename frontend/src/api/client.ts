const BASE = '/api/v1'

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
  // 204 No Content (e.g. DELETE) has an empty body — nothing to parse
  if (res.status === 204) return undefined as T
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
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  delete: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'DELETE',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    }),
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

// ---------- models ----------

export interface ModelOut {
  id: string
  name: string
  classes: Record<number, string>
  task: string
  created_at: string
  source: 'upload' | 'official' | 'trained'
}

export interface OfficialModel {
  name: string
  default: boolean
}

export const patchModel = (id: string, name: string) =>
  api.patch<ModelOut>(`/models/${id}`, { name })

export const modelDownloadUrl = (id: string) => `${BASE}/models/${id}/download`

// ---------- projects ----------

export interface ProjectOut {
  id: string
  name: string
  created_at: string
}

export interface StatsOut {
  images: number
  labeled: number // label file exists (may be empty = a "no class" negative)
  reviewed: number
  classes: { id: number; name: string; sources: string[] }[]
}

export interface LabelBox {
  id?: string | null
  cls: number
  xyxy_n: [number, number, number, number]
  score?: number | null
  status?: string | null
  reason?: string | null
  sources?: { model: string; score: number }[] | null
}

export interface ImageItem {
  name: string
  stem: string
  thumb: string
  url: string
  labeled: boolean
  reviewed: boolean
  boxes: LabelBox[]
  created_at: number
}

export interface ImagePage {
  total: number
  page: number
  size: number
  items: ImageItem[]
}

export interface ImageQuery {
  labeled?: boolean
  reviewed?: boolean
  cls?: number
  sort?: 'created' | 'name'
  order?: 'asc' | 'desc'
  q?: string
  page?: number
  size?: number
}

export function imagesQueryString(query: ImageQuery): string {
  const params = new URLSearchParams()
  if (query.labeled != null) params.set('labeled', String(query.labeled))
  if (query.reviewed != null) params.set('reviewed', String(query.reviewed))
  if (query.cls != null) params.set('cls', String(query.cls))
  if (query.sort) params.set('sort', query.sort)
  if (query.order) params.set('order', query.order)
  if (query.q) params.set('q', query.q)
  if (query.page) params.set('page', String(query.page))
  if (query.size) params.set('size', String(query.size))
  return params.toString()
}

export const listImages = (projectId: string, query: ImageQuery) =>
  api.get<ImagePage>(`/projects/${projectId}/images?${imagesQueryString(query)}`)

export const listImageNames = (projectId: string, query: ImageQuery) =>
  api.get<{ total: number; names: string[] }>(
    `/projects/${projectId}/images?${imagesQueryString(query)}&names_only=true`,
  )

export const deleteImages = (projectId: string, names: string[]) =>
  api.delete<{ deleted: number }>(`/projects/${projectId}/images`, { names })

export const putReviewed = (projectId: string, stem: string, reviewed: boolean) =>
  api.put<{ ok: boolean; reviewed: boolean }>(
    `/projects/${projectId}/images/${encodeURIComponent(stem)}/reviewed`,
    { reviewed },
  )

// ---------- labels ----------

export interface LabelDetail {
  stem: string
  name: string
  image_url: string
  boxes: LabelBox[]
  classes: { id: number; name: string; sources: string[] }[]
  reviewed: boolean
}

export const getLabels = (projectId: string, stem: string) =>
  api.get<LabelDetail>(`/projects/${projectId}/labels/${encodeURIComponent(stem)}`)

export const putLabels = (projectId: string, stem: string, boxes: LabelBox[]) =>
  api.put(`/projects/${projectId}/labels/${encodeURIComponent(stem)}`, { boxes })

// ---------- classes ----------

export interface ProjectClass {
  id: number
  name: string
  sources: string[]
}

export const listClasses = (projectId: string) =>
  api.get<ProjectClass[]>(`/projects/${projectId}/classes`)

export const addClass = (projectId: string, name: string) =>
  api.post<ProjectClass>(`/projects/${projectId}/classes`, { name })

export const renameClass = (projectId: string, classId: number, name: string) =>
  api.patch<ProjectClass>(`/projects/${projectId}/classes/${classId}`, { name })

export const deleteClass = (projectId: string, classId: number) =>
  api.delete<{ ok: boolean; removed_boxes: number; classes: ProjectClass[] }>(
    `/projects/${projectId}/classes/${classId}`,
  )

// ---------- auto-label jobs ----------

export interface JobCreate {
  model_ids: string[]
  conf: number
  iou_wbf: number
  imgsz: number
  batch_size: number
  names: string[] | null
}

export interface JobOut {
  id: string
  project_id: string
  status: string
  config: Record<string, unknown>
  result: {
    total: number
    labeled: number
    boxes: number
  } | null
  error: string | null
  created_at: string
}

export interface JobProgressEvent {
  phase: 'inference' | 'done' | 'error' | 'cancelled'
  done?: number
  total?: number
  labeled?: number
  boxes?: number
  msg?: string
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

// ---------- exports ----------

export interface ExportCreate {
  kind: 'yolo' | 'images'
  val_split?: number
  seed?: number
  names?: string[] | null
}

export interface ExportOut {
  id: string
  name: string
  kind: 'yolo' | 'images'
  created_at: string
  val_split: number
  seed: number
  train: number
  val: number
  count: number
  classes: number
  size_bytes: number
}

export const createExport = (projectId: string, body: ExportCreate) =>
  api.post<ExportOut>(`/projects/${projectId}/exports`, body)

export const listExports = (projectId: string) =>
  api.get<ExportOut[]>(`/projects/${projectId}/exports`)

export const renameExport = (projectId: string, exportId: string, name: string) =>
  api.patch<ExportOut>(`/projects/${projectId}/exports/${exportId}`, { name })

export const deleteExport = (projectId: string, exportId: string) =>
  api.delete<void>(`/projects/${projectId}/exports/${exportId}`)

export const exportDownloadUrl = (projectId: string, exportId: string) =>
  `${BASE}/projects/${projectId}/exports/${exportId}/download`

// ---------- training ----------

export interface TrainDataset {
  dataset: string // token: "export:{pid}:{eid}" | "upload:{uid}"
  name: string
  source: 'export' | 'upload'
  train: number
  val: number
  classes: number
  created_at: string
}

export const uploadTrainDataset = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api.upload<TrainDataset & { id: string }>('/training/datasets', form)
}

export interface TrainParams {
  epochs: number
  imgsz: number
  batch: number
  patience?: number
  lr0?: number | null
  optimizer?: string | null
}

export interface RunCreate {
  name?: string | null
  project_id?: string | null
  dataset: string
  base_model_id: string
  device?: string | null
  params: TrainParams
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
  phase: 'preparing' | 'start' | 'epoch_start' | 'epoch' | 'done' | 'error' | 'cancelled'
  epoch?: number
  epochs?: number
  metrics?: Record<string, number>
  msg?: string
}

export function subscribeTrainEvents(
  runId: string,
  onEvent: (ev: TrainEpochEvent) => void,
): () => void {
  const source = new EventSource(`${BASE}/training/runs/${runId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as TrainEpochEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') {
      source.close()
    }
  })
  return () => source.close()
}

// full per-epoch metrics from ultralytics results.csv (train+val loss, mAP, P/R, lr, time)
export type TrainResultRow = Record<string, number | string>

export const getRunResults = (runId: string) =>
  api.get<TrainResultRow[]>(`/training/runs/${runId}/results`)

export interface PerClassRow {
  cls: number
  name: string
  instances: number | null
  precision: number
  recall: number
  mAP50: number
  'mAP50-95': number
}

export const getRunPerClass = (runId: string) =>
  api.get<PerClassRow[]>(`/training/runs/${runId}/per-class`)

export interface PerClassEpoch {
  epoch: number
  metrics: PerClassRow[]
}

export const getRunPerClassHistory = (runId: string) =>
  api.get<PerClassEpoch[]>(`/training/runs/${runId}/per-class-history`)

// ---------- videos ----------

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

// ---------- test / inference (playground — no DB writes) ----------

export interface PredictBox {
  cls: number
  name: string
  score: number
  xyxyn: [number, number, number, number]
  model_ids: string[]
  agree: number
}

export interface PredictResponse {
  task: string
  names: Record<number, string>
  boxes: PredictBox[]
  device: string
  floor: number
}

export interface ResidentModel {
  model_id: string
  device?: string | null
  task?: string | null
  classes?: Record<number, string> | null
  weight_mb?: number | null
  loaded_at?: number | null
}

export interface ResourceInfo {
  accelerator: 'cuda' | 'mps' | 'cpu'
  device_label: string
  ram_total_mb?: number | null
  ram_available_mb?: number | null
  ram_used_mb?: number | null
  vram_total_mb?: number | null
  vram_used_mb?: number | null
  vram_free_mb?: number | null
  training_active: boolean
  resident_models: number
  warning?: string | null
}

export interface PredictParams {
  conf: number
  iou_wbf: number
  imgsz: number
  device?: string | null
}

export const getResources = () => api.get<ResourceInfo>('/system/resources')

export const listResidents = () => api.get<ResidentModel[]>('/predict/residents')

export const loadResident = (modelId: string, projectId: string, device?: string) =>
  api.post<ResidentModel>(
    `/predict/residents/${modelId}?project_id=${projectId}${device ? `&device=${device}` : ''}`,
  )

export const unloadResident = (modelId: string) => api.delete<void>(`/predict/residents/${modelId}`)

export function predict(opts: {
  modelIds: string[]
  projectId: string
  params: PredictParams
  file?: File
  imageProjectId?: string
  imageName?: string
}): Promise<PredictResponse> {
  const form = new FormData()
  form.append('model_ids', opts.modelIds.join(','))
  form.append('project_id', opts.projectId)
  form.append('conf', String(opts.params.conf))
  form.append('iou_wbf', String(opts.params.iou_wbf))
  form.append('imgsz', String(opts.params.imgsz))
  if (opts.params.device) form.append('device', opts.params.device)
  if (opts.file) form.append('file', opts.file)
  if (opts.imageProjectId) form.append('image_project_id', opts.imageProjectId)
  if (opts.imageName) form.append('image_name', opts.imageName)
  return api.upload<PredictResponse>('/predict', form)
}

// ---------- test: video frame scrub ----------

export interface VideoUpload {
  video_id: string
  frame_count: number
}

export function uploadTestVideo(file: File): Promise<VideoUpload> {
  const form = new FormData()
  form.append('file', file)
  return api.upload<VideoUpload>('/predict/video', form)
}

export const videoFrameUrl = (videoId: string, idx: number) =>
  `${BASE}/predict/video/${videoId}/frame/${idx}`

export function predictVideoFrame(opts: {
  videoId: string
  idx: number
  modelIds: string[]
  projectId: string
  params: PredictParams
}): Promise<PredictResponse> {
  const form = new FormData()
  form.append('model_ids', opts.modelIds.join(','))
  form.append('project_id', opts.projectId)
  form.append('conf', String(opts.params.conf))
  form.append('iou_wbf', String(opts.params.iou_wbf))
  form.append('imgsz', String(opts.params.imgsz))
  if (opts.params.device) form.append('device', opts.params.device)
  return api.upload<PredictResponse>(`/predict/video/${opts.videoId}/frame/${opts.idx}`, form)
}

// 데이터셋 — 학습실의 작업 단위.
//
// 데이터셋 하나가 이미지·라벨·클래스·검수 상태·분할을 **자기가 전부 갖는다.**
// 데이터셋끼리는 아무것도 공유하지 않으므로 삭제도 통째로다.
import { BASE, api, xhrUpload } from './http'
import type { UploadHandlers } from './http'
import type { ImageItem, LabelBox } from './projects'


export interface DatasetOut {
  id: string
  name: string
  created_at: string
  // 수치는 서버가 파일에서 그때 센다 — 저장된 값이 아니라 항상 실제와 맞는다
  images: number
  unreviewed: number
  reviewed: number
  unassigned: number // 검수완료인데 아직 train/val/test 어디에도 없는 것
  train: number
  val: number
  test: number
}

export const listDatasets = (projectId: string) =>
  api.get<DatasetOut[]>(`/projects/${projectId}/datasets`)

export const getDataset = (projectId: string, datasetId: string) =>
  api.get<DatasetOut>(`/projects/${projectId}/datasets/${datasetId}`)

export const createDataset = (projectId: string, name: string) =>
  api.post<DatasetOut>(`/projects/${projectId}/datasets`, { name })

export const renameDataset = (projectId: string, datasetId: string, name: string) =>
  api.patch<DatasetOut>(`/projects/${projectId}/datasets/${datasetId}`, { name })

export const deleteDataset = (projectId: string, datasetId: string) =>
  api.delete<void>(`/projects/${projectId}/datasets/${datasetId}`)

// ---------- 출처 ----------
//
// 영상은 프레임을 뽑고 버리므로 **이 목록이 유일한 기록**이다. 같은 이름이 두 번
// 들어왔을 때 어느 쪽이 `(2)` 인지도 여기서 본다.

export interface DatasetSource {
  id: string
  kind: 'video' | 'dataset'
  filename: string
  /** 영상만. 이 출처에서 나온 프레임의 이름 앞부분 (`경기 영상` · `경기 영상 (2)`) */
  stem: string | null
  at: string
  status: 'running' | 'done' | 'error' | 'cancelled'
  // 영상
  frames?: number | null
  params?: Record<string, number | boolean | null>
  // zip
  images?: number
  labeled?: number
  reviewed?: boolean
  assigned?: number
}

export const listDatasetSources = (projectId: string, datasetId: string) =>
  api.get<DatasetSource[]>(`/projects/${projectId}/datasets/${datasetId}/sources`)

// ---------- 이미지 ----------

/** `ImageItem` 에 분할만 얹은 것 — 그래서 `ImageGrid` 를 그대로 쓴다. */
export interface DatasetImage extends ImageItem {
  split: 'train' | 'val' | 'test' | null
}

export interface DatasetImagesPage {
  total: number
  page: number
  size: number
  items: DatasetImage[]
  names?: string[]
}

/** `reviewed` 가 1급 필터고 `split` 은 검수완료 안의 갈래다. `split=none` 은 미할당. */
export interface DatasetImageQuery {
  reviewed?: boolean
  split?: 'train' | 'val' | 'test' | 'none'
  labeled?: boolean
  /** 클래스 id, 또는 **-1 = 클래스 없음**(라벨은 있는데 박스가 없는 네거티브) */
  cls?: number
  sort?: 'created' | 'name'
  order?: 'asc' | 'desc'
  q?: string
  page?: number
  size?: number
  names_only?: boolean
}

export function listDatasetImages(
  projectId: string,
  datasetId: string,
  query: DatasetImageQuery = {},
) {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(query)) {
    if (v !== undefined && v !== '') p.set(k, String(v))
  }
  const qs = p.toString()
  return api.get<DatasetImagesPage>(
    `/projects/${projectId}/datasets/${datasetId}/images${qs ? `?${qs}` : ''}`,
  )
}

export const setImageReviewed = (
  projectId: string,
  datasetId: string,
  stem: string,
  reviewed: boolean,
) =>
  api.put<{ ok: boolean; reviewed: boolean }>(
    `/projects/${projectId}/datasets/${datasetId}/images/${stem}/reviewed`,
    { reviewed },
  )

export const deleteDatasetImages = (projectId: string, datasetId: string, names: string[]) =>
  api.delete<{ deleted: number }>(`/projects/${projectId}/datasets/${datasetId}/images`, {
    names,
  })

// ---------- 분할 · 내보내기 ----------

export interface SplitParams {
  train?: number
  val?: number
  test?: number
  seed?: number
  /** 전부 다시 섞기 — **명시할 때만**. 기본은 미할당만 나눈다 */
  reassign_all?: boolean
}

export const splitDataset = (projectId: string, datasetId: string, params: SplitParams = {}) =>
  api.post<DatasetOut>(`/projects/${projectId}/datasets/${datasetId}/split`, params)

export type ExportKind = 'train' | 'test' | 'all'

export interface ExportResult {
  id: string
  kind: ExportKind
  count: number
  train: number
  val: number
  test: number
  classes: number
  linked: number // 하드링크로 들어간 수 — 복사가 아니라는 증거
  copied: number
  size_bytes: number
}

export const exportDataset = (
  projectId: string,
  datasetId: string,
  kind: ExportKind,
  classIds?: number[],
) =>
  api.post<ExportResult>(`/projects/${projectId}/datasets/${datasetId}/export`, {
    kind,
    class_ids: classIds ?? null,
  })

export const datasetExportUrl = (projectId: string, datasetId: string, exportId: string) =>
  `${BASE}/projects/${projectId}/datasets/${datasetId}/export/${exportId}/download`

// ---------- 라벨 · 클래스 ----------
//
// 클래스는 **이 데이터셋 것**이다. 같은 이미지를 다른 데이터셋에 넣었더라도 여기서
// 보는 목록은 이 데이터셋의 classes.json 이다.

export interface DatasetLabels {
  stem: string
  name: string
  image_url: string
  boxes: LabelBox[]
  classes: { id: number; name: string }[]
  reviewed: boolean
}

export const getDatasetLabels = (projectId: string, datasetId: string, stem: string) =>
  api.get<DatasetLabels>(`/projects/${projectId}/datasets/${datasetId}/labels/${stem}`)

export const putDatasetLabels = (
  projectId: string,
  datasetId: string,
  stem: string,
  boxes: unknown[],
) =>
  api.put<{ ok: boolean; count: number }>(
    `/projects/${projectId}/datasets/${datasetId}/labels/${stem}`,
    { boxes },
  )

/** 클래스는 **데이터셋마다** 따로다 — 자리도 그 데이터셋의 `classes.json` 이다. */
export interface DatasetClass {
  id: number
  name: string
  /** 이 데이터셋의 라벨에서 이 클래스를 쓰는 박스 수. 지우기 전에 보여 준다. */
  boxes: number
}

export const listDatasetClasses = (projectId: string, datasetId: string) =>
  api.get<DatasetClass[]>(`/projects/${projectId}/datasets/${datasetId}/classes`)

export const addDatasetClass = (projectId: string, datasetId: string, name: string) =>
  api.post<{ id: number; name: string }>(
    `/projects/${projectId}/datasets/${datasetId}/classes`,
    { name },
  )

export const renameDatasetClass = (
  projectId: string,
  datasetId: string,
  classId: number,
  name: string,
) =>
  api.patch<{ id: number; name: string }>(
    `/projects/${projectId}/datasets/${datasetId}/classes/${classId}`,
    { name },
  )

export const deleteDatasetClass = (projectId: string, datasetId: string, classId: number) =>
  api.delete<{ ok: boolean; removed_boxes: number; classes: { id: number; name: string }[] }>(
    `/projects/${projectId}/datasets/${datasetId}/classes/${classId}`,
  )

// ---------- 가져오기 ----------
//
// 두 방식 모두 결과는 **미검수**로 들어온다. 동영상은 프레임을 얻는 수단일 뿐이라
// 추출이 끝나면 서버가 지운다.

// 타일링은 여기 없다 — 타일은 **학습 전처리**의 선택이지 데이터를 모으는 일이 아니다.
// 작은 객체를 잡는 문제는 오토라벨링의 `tiled` 추론이 맡는다.
export interface VideoImportParams {
  target_fps: number
  max_frames: number
  start_sec: number
  end_sec: number | null
  dedup: boolean
  dedup_threshold: number
}

export interface ImportProgressEvent {
  phase: 'start' | 'extract' | 'done' | 'error' | 'cancelled'
  saved?: number
  scanned?: number
  total_frames?: number
  skipped_dup?: number
  msg?: string
}

const importBase = (projectId: string, datasetId: string) =>
  `/projects/${projectId}/datasets/${datasetId}/import`

export function importVideo(
  projectId: string,
  datasetId: string,
  file: File,
  params: VideoImportParams,
  handlers: UploadHandlers = {},
) {
  const form = new FormData()
  form.append('file', file)
  form.append('target_fps', String(params.target_fps))
  form.append('max_frames', String(params.max_frames))
  form.append('start_sec', String(params.start_sec))
  if (params.end_sec != null) form.append('end_sec', String(params.end_sec))
  form.append('dedup', String(params.dedup))
  form.append('dedup_threshold', String(params.dedup_threshold))
  return xhrUpload<{ job_id: string; filename: string; status: string }>(
    `${importBase(projectId, datasetId)}/video`,
    form,
    handlers,
  )
}

export interface ZipImportResult {
  images: number
  labeled: number
  classes: number
  reviewed: boolean
  /** 폴더 구조에서 train/val/test 로 배정된 수 (검증됨으로 가져왔을 때만) */
  assigned: number
}

/** YOLO zip 가져오기. 잡이 아니라 그 자리에서 끝나고 결과를 돌려준다.
 *
 *  `reviewed` 는 "이미 사람이 검증한 데이터"라는 뜻이다. 참이면 검수완료로 들어오고
 *  zip 의 train/val/test 폴더가 그대로 분할이 된다.
 */
export function importYoloZip(
  projectId: string,
  datasetId: string,
  file: File,
  reviewed: boolean,
  handlers: UploadHandlers = {},
) {
  const form = new FormData()
  form.append('file', file)
  form.append('reviewed', String(reviewed))
  return xhrUpload<ZipImportResult>(
    `${importBase(projectId, datasetId)}/dataset`,
    form,
    handlers,
  )
}

export const cancelImport = (projectId: string, datasetId: string, jobId: string) =>
  api.post(`${importBase(projectId, datasetId)}/${jobId}/cancel`)

export function subscribeImportEvents(
  projectId: string,
  datasetId: string,
  jobId: string,
  onEvent: (ev: ImportProgressEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}${importBase(projectId, datasetId)}/${jobId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as ImportProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') source.close()
  })
  source.onerror = () => {
    // EventSource 는 일시적 끊김을 스스로 재시도한다 — CLOSED 만 진짜 실패다
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}

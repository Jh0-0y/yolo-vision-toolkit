// 모델 비교 — 정답 라벨 대비 채점.

import { BASE, api } from '../http'

/** 잡을 시작하는 엔드포인트의 공통 응답 — 진행률은 SSE 로 따로 붙는다. */
export interface TestJobStart {
  job_id: string
}

// ---------- test: model comparison (score models vs labeled ground truth) ----------

export interface ClassMetric {
  cls: number
  name: string
  tp: number
  fp: number
  fn: number
  gt: number
  pred: number
  precision: number
  recall: number
  f1: number
  ap50?: number
  ap?: number
}

export interface CompareOverall {
  tp: number
  fp: number
  fn: number
  precision: number
  recall: number
  f1: number
}

export interface CompareBox {
  cls: number
  name: string
  xyxyn: [number, number, number, number]
  score?: number
}

/** 검출기 엔트리 — 모델과 그 모델의 추론 방식. `conf` 는 없다: 벤치마크는
 *  모든 엔트리를 **하나의 전역 동작점**에서 채점해야 비교가 성립한다. */
export interface BenchmarkEntry {
  model_id: string
  mode: 'full' | 'tiled'
  imgsz: number
  tile_size: number
  stride: number
  merge_iou: number
  border_margin_px: number
}

export interface BenchmarkStartBody {
  dataset: string // "dataset:{projectId}:{datasetId}"
  entries: BenchmarkEntry[]
  conf: number
  iou: number
  device?: string | null
}

export interface BenchmarkOut {
  id: string
  created_at: string
  dataset_name: string
  dataset: string
  entries: number
  conf: number
  iou: number
  status: 'running' | 'done' | 'error' | 'cancelled'
  error: string | null
}

/** 결과는 **엔트리 단위**로 키를 잡는다 — 같은 모델을 방식만 바꿔 두 번 넣을 수 있다. */
export interface CompareEntryResult {
  entry_id: string
  model_id: string
  name: string
  mode: 'full' | 'tiled'
  /** 화면의 엔트리 제목에 크기를 붙이려고 함께 온다 — `full` 은 `imgsz`,
   *  `tiled` 는 `tile_size` 가 그 엔트리를 규정하는 숫자다.
   *
   *  **선택이다.** result.json 은 디스크에 남는 영속 포맷이라, 이 두 필드가
   *  생기기 전에 돌린 런의 결과에는 키가 아예 없다. */
  imgsz?: number
  tile_size?: number
  overall: CompareOverall
  per_class: ClassMetric[]
  detections: number
  map50: number
  map: number
}

export interface CompareImage {
  stem: string
  name: string
  url: string
  gt_boxes: CompareBox[]
  per_entry: { entry_id: string; pred_boxes: CompareBox[] }[]
}

export interface CompareResult {
  per_entry: CompareEntryResult[]
  images: CompareImage[]
  image_count: number
  conf: number
  iou: number
  warning?: string | null
}

export interface CompareProgress {
  phase: 'start' | 'analyze' | 'done' | 'error' | 'cancelled'
  done?: number
  total?: number
  msg?: string
}

export const startBenchmark = (body: BenchmarkStartBody) =>
  api.post<TestJobStart>('/predict/benchmarks', body)

export const listBenchmarks = (projectId: string) =>
  api.get<BenchmarkOut[]>(`/predict/benchmarks?project_id=${projectId}`)

export const deleteBenchmark = (projectId: string, benchId: string) =>
  api.delete<void>(`/predict/benchmarks/${benchId}?project_id=${projectId}`)

export const getBenchmarkResult = (projectId: string, benchId: string) =>
  api.get<CompareResult>(`/predict/benchmarks/${benchId}/result?project_id=${projectId}`)

export function subscribeBenchmarkEvents(
  benchId: string,
  onEvent: (ev: CompareProgress) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}/predict/benchmarks/${benchId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as CompareProgress
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') source.close()
  })
  source.onerror = () => {
    // EventSource 는 일시적 끊김을 스스로 재시도한다 — CLOSED 만 진짜 실패다
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}

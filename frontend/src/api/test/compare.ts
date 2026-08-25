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

/** 곡선 하나 — 클래스 하나의 점 목록. `points` 는 `[x, y]` 쌍으로,
 *  PR 곡선은 `[recall, precision]`, F1-conf 곡선은 `[conf, f1]` 이다. */
export interface CurvePoints {
  cls: number
  name: string
  points: [number, number][]
}

/** conf 를 하나로 고정해 다시 채점한 스냅샷 — 화면의 conf 슬라이더가 이 목록 위를 움직인다. */
export interface OperatingPoint {
  conf: number
  overall: CompareOverall
  per_class: ClassMetric[]
  /** 행이 실제, 열이 예측. 마지막 행·열은 background 이고,
   *  background×background 칸은 뜻이 없어 `null` 이다. */
  confusion: {
    labels: string[]
    rows: (number | null)[][]
  }
}

/** 객체 크기 구간별 성적. 0.5:0.95 평균은 메모리 비용이 너무 커서 싣지 않는다 —
 *  이 지표가 답하려는 질문에는 AP@0.5 로 충분하다. */
export interface SizeMetric {
  ap50: number
  gt: number
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
  /** 여기부터는 전부 **선택**이다 — result.json 은 디스크에 남는 영속 포맷이라
   *  이 지표들이 생기기 전에 돌린 런의 결과에는 키가 아예 없다. */
  ap50?: number | null
  ap75?: number | null
  /** 정답이 하나도 없는 구간은 아예 빠진다 — 그래서 세 키가 다 있다고 볼 수 없다. */
  by_size?: Partial<Record<'small' | 'medium' | 'large', SizeMetric>>
  curves?: {
    pr: CurvePoints[]
    f1_conf: CurvePoints[]
    /** F1 이 가장 높은 지점. `cls`·`name` 은 그 F1 이 어느 클래스의 것인지 —
     *  다중 클래스에서 이름 없는 conf 하나만 보여 주면 뜻이 흐려진다. */
    best_f1: {
      value: number
      conf: number
      cls: number
      name: string
    } | null
  }
  operating_points?: OperatingPoint[]
  speed?: {
    ms_median: number
    ms_p95: number
    fps: number | null
  } | null
  model?: {
    size_bytes: number | null
    params: number | null
  } | null
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

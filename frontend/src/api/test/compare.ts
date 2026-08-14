// 모델 비교 — 정답 라벨 대비 채점.

import { BASE, api } from '../http'
import type { PredictParams } from './predict'

/** 잡을 시작하는 엔드포인트의 공통 응답 — 진행률은 SSE 로 따로 붙는다. */
export interface TestJobStart {
  job_id: string
  status: string
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

export interface CompareModelResult {
  model_id: string
  name: string
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
  per_model: { model_id: string; pred_boxes: CompareBox[] }[]
}

export interface CompareResult {
  per_model: CompareModelResult[]
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

export function startCompare(opts: {
  projectId: string
  modelIds: string[]
  file: File // YOLO test-set zip: images/ + labels/ + data.yaml
  params: PredictParams
}): Promise<TestJobStart> {
  const form = new FormData()
  form.append('project_id', opts.projectId)
  form.append('model_ids', opts.modelIds.join(','))
  form.append('file', opts.file)
  form.append('conf', String(opts.params.conf))
  form.append('iou', String(opts.params.iou_wbf)) // reused as pred↔GT match IoU
  form.append('imgsz', String(opts.params.imgsz))
  if (opts.params.device) form.append('device', opts.params.device)
  return api.upload<TestJobStart>('/predict/compare', form)
}

export function subscribeCompareEvents(jobId: string, onEvent: (ev: CompareProgress) => void): () => void {
  const source = new EventSource(`${BASE}/predict/compare/${jobId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as CompareProgress
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') source.close()
  })
  return () => source.close()
}

export const getCompareResult = (jobId: string) =>
  api.get<CompareResult>(`/predict/compare/${jobId}/result`)

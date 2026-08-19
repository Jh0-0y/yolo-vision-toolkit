// 학습 — 실행 · 실시간 지표 · 산출물.
//
// 학습은 **데이터셋을 직접 먹는다.** 따로 올릴 것이 없으므로 여기에 업로드는 없다.
import { BASE, api } from './http'

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
  dataset: string // 토큰: "dataset:{project_id}:{dataset_id}"
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

// direct-download URL for the raw results.csv (per-epoch training metrics)
export const runResultsCsvUrl = (runId: string) => `${BASE}/training/runs/${runId}/results.csv`

// direct-download URL for ultralytics args.yaml (fully-resolved training config)
export const runArgsYamlUrl = (runId: string) => `${BASE}/training/runs/${runId}/args.yaml`

export const getRunLog = (runId: string) =>
  api.get<{ text: string; truncated: boolean }>(`/training/runs/${runId}/log`)

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

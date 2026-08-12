// 학습 — 데이터셋 업로드/선택 · 실행 · 실시간 지표 · 산출물.
import { BASE, api, xhrUpload } from './http'
import type { UploadHandlers } from './http'


export interface TrainDataset {
  dataset: string // token: "export:{pid}:{eid}" | "upload:{uid}"
  name: string
  source: 'export' | 'upload'
  train: number
  val: number
  classes: number
  created_at: string
  auto_delete?: boolean // uploads only: self-delete after a successful run
}

export function uploadTrainDataset(
  file: File,
  handlers: UploadHandlers = {},
): Promise<TrainDataset & { id: string }> {
  const form = new FormData()
  form.append('file', file)
  return xhrUpload<TrainDataset & { id: string }>('/training/datasets', form, handlers)
}

// dataset_id is the "{uid}" part of an "upload:{uid}" token
export const deleteTrainDataset = (datasetId: string) =>
  api.delete<void>(`/training/datasets/${datasetId}`)

// toggle self-delete-after-training on an uploaded dataset
export const patchTrainDataset = (datasetId: string, autoDelete: boolean) =>
  api.patch<{ dataset: string; auto_delete: boolean }>(`/training/datasets/${datasetId}`, {
    auto_delete: autoDelete,
  })

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

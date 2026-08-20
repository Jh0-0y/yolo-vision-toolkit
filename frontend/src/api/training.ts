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

/** 학습 전처리 타일링. 끄면 지금까지와 완전히 같다. */
export interface TilingParams {
  enabled: boolean
  tile_size: number
  stride: number
  min_visibility: number
  /** 포지티브 타일 수 대비 비율 (0.1 = 10%). **분할마다 따로** 적용된다 */
  negative_ratio: number
  keep_all_negatives: boolean
  seed: number
}

export const DEFAULT_TILING: TilingParams = {
  enabled: false,
  tile_size: 640,
  stride: 480,
  min_visibility: 0.6,
  negative_ratio: 0.1,
  keep_all_negatives: false,
  seed: 0,
}

export interface SplitPreview {
  split: string
  positive: number
  /** 라벨 없는 프레임에서 나온 네거티브 — 비율과 무관하게 전부 담긴다 */
  hard: number
  /** 라벨 있는 프레임의 빈 타일 — 목표까지만 담긴다 */
  incidental: number
  negative_kept: number
  /** 박스가 걸쳤으나 가시비율 미달 — 어디에도 안 들어간다 */
  excluded: number
  total: number
  /** 하드 네거티브만으로 목표를 넘었나 */
  overshoot: boolean
}

export interface TilingPreview {
  splits: SplitPreview[]
}

export const previewTiling = (dataset: string, tiling: TilingParams) =>
  api.post<TilingPreview>('/training/tiling/preview', { dataset, tiling })

export interface RunCreate {
  name?: string | null
  project_id?: string | null
  dataset: string // 토큰: "dataset:{project_id}:{dataset_id}"
  base_model_id: string
  device?: string | null
  params: TrainParams
  tiling?: TilingParams
}

export interface TrainRunOut {
  id: string
  name: string
  status: string
  dataset_path: string
  base_model_id: string
  base_model_name: string | null
  params: Record<string, number | string>
  /** config.json 의 타일링 노브 — 하드링크 런이면 null */
  tiling: TilingParams | null
  metrics: Record<string, number> | null
  error: string | null
  created_at: string
  finished_at: string | null
}

export interface TrainEpochEvent {
  phase:
    | 'preparing'
    | 'staging'
    | 'tiling'
    | 'materialized'
    | 'start'
    | 'epoch_start'
    | 'epoch'
    | 'done'
    | 'error'
    | 'cancelled'
  epoch?: number
  epochs?: number
  metrics?: Record<string, number>
  msg?: string
  /** phase 'tiling' 전용 — 지금까지 자른 타일 수 */
  done?: number
  /** phase 'tiling' 전용 — 전체 타일 수 */
  total?: number
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

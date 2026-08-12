// 단발 추론과 리소스 조회 (플레이그라운드 — DB에 쓰지 않는다).

import { api } from '../http'


export interface PredictBox {
  cls: number
  name: string
  score: number
  xyxyn: [number, number, number, number]
  model_ids: string[]
  agree: number
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
  conf?: number // omit to use the backend default
  iou_wbf: number
  imgsz: number
  device?: string | null
}

export const getResources = () => api.get<ResourceInfo>('/system/resources')

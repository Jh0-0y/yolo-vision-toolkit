// 모델 레지스트리 — 공식 모델 · 업로드 · 학습결과 등록.
import { BASE, api } from './http'


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

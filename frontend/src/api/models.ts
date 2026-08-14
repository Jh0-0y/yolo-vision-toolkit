// 모델 레지스트리 — 공식 모델 · 업로드 · 학습결과 등록.
import { BASE, api } from './http'


export interface ModelOut {
  id: string
  name: string
  classes: Record<number, string>
  task: string
  created_at: string
  source: 'upload' | 'official' | 'trained'
  // 어느 프로젝트에서 나온 모델인지 — null 은 공유 풀. 연구실은 전체 목록에서
  // 고르므로 출처를 묶어 보여줄 때 쓴다.
  project_id: string | null
  project_name: string | null
}

/** 전체 모델(공유 풀 + 모든 프로젝트). 프로젝트에 속하지 않는 연구실이 쓴다. */
export const listAllModels = () => api.get<ModelOut[]>('/models')

export interface OfficialModel {
  name: string
  default: boolean
}

export const patchModel = (id: string, name: string) =>
  api.patch<ModelOut>(`/models/${id}`, { name })

export const modelDownloadUrl = (id: string) => `${BASE}/models/${id}/download`

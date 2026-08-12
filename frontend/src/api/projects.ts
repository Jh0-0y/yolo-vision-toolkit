// 프로젝트와 그 안의 이미지 — 목록 · 필터 · 통계 · 검수 플래그.
import { api } from './http'


export interface ProjectOut {
  id: string
  name: string
  created_at: string
}

export interface StatsOut {
  images: number
  labeled: number // label file exists (may be empty = a "no class" negative)
  reviewed: number
  classes: { id: number; name: string; sources: string[] }[]
}

export interface LabelBox {
  id?: string | null
  cls: number
  xyxy_n: [number, number, number, number]
  score?: number | null
  status?: string | null
  reason?: string | null
  sources?: { model: string; score: number }[] | null
}

export interface ImageItem {
  name: string
  stem: string
  thumb: string
  url: string
  labeled: boolean
  reviewed: boolean
  boxes: LabelBox[]
  created_at: number
}

export interface ImagePage {
  total: number
  page: number
  size: number
  items: ImageItem[]
}

export interface ImageQuery {
  labeled?: boolean
  reviewed?: boolean
  cls?: number
  sort?: 'created' | 'name'
  order?: 'asc' | 'desc'
  q?: string
  page?: number
  size?: number
}

export function imagesQueryString(query: ImageQuery): string {
  const params = new URLSearchParams()
  if (query.labeled != null) params.set('labeled', String(query.labeled))
  if (query.reviewed != null) params.set('reviewed', String(query.reviewed))
  if (query.cls != null) params.set('cls', String(query.cls))
  if (query.sort) params.set('sort', query.sort)
  if (query.order) params.set('order', query.order)
  if (query.q) params.set('q', query.q)
  if (query.page) params.set('page', String(query.page))
  if (query.size) params.set('size', String(query.size))
  return params.toString()
}

export const listImages = (projectId: string, query: ImageQuery) =>
  api.get<ImagePage>(`/projects/${projectId}/images?${imagesQueryString(query)}`)

export const listImageNames = (projectId: string, query: ImageQuery) =>
  api.get<{ total: number; names: string[] }>(
    `/projects/${projectId}/images?${imagesQueryString(query)}&names_only=true`,
  )

export const deleteImages = (projectId: string, names: string[]) =>
  api.delete<{ deleted: number }>(`/projects/${projectId}/images`, { names })

export const putReviewed = (projectId: string, stem: string, reviewed: boolean) =>
  api.put<{ ok: boolean; reviewed: boolean }>(
    `/projects/${projectId}/images/${encodeURIComponent(stem)}/reviewed`,
    { reviewed },
  )

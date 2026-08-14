// 연구실(Lab) — **하나뿐이다.** 그 하나와 영상 아카이브.
//
// 학습실 프로젝트(`projects.ts`)와 달리 목록이 아니다 — 연구는 하나고, 종목이
// 늘어나면 파이프라인부터 다른 연구실이 코드로 들어온다. 그래서 경로에 id 가 없다.
import { BASE, api, xhrUpload } from './http'
import type { UploadHandlers } from './http'


export interface LabOut {
  id: string
  name: string
  description: string // 무슨 연구인지 — 백엔드가 코드로 갖는다
  created_at: string
  video_count: number
  run_count: number
}

export interface LabVideoOut {
  id: string
  name: string // 사용자에게 보이는 이름 (기본값 = 업로드한 파일명)
  filename: string
  ext: string
  size_bytes: number
  width: number
  height: number
  fps: number
  duration_ms: number
  created_at: string
  run_count: number // 이 영상으로 돌린 크롭 런 수
}

export const getLab = () => api.get<LabOut>('/lab')

// ---------- 영상 아카이브 ----------

export const listLabVideos = () => api.get<LabVideoOut[]>(`/lab/videos`)

// 원본은 기가 단위라 진행률이 필요하다 — fetch 로는 못 재므로 XHR 을 쓴다.
export function uploadLabVideo(file: File, handlers: UploadHandlers = {}) {
  const form = new FormData()
  form.append('file', file)
  return xhrUpload<LabVideoOut>(`/lab/videos`, form, handlers)
}

export const renameLabVideo = (videoId: string, name: string) =>
  api.patch<LabVideoOut>(`/lab/videos/${videoId}`, { name })

export const deleteLabVideo = (videoId: string) =>
  api.delete<void>(`/lab/videos/${videoId}`)

export const labVideoUrl = (videoId: string) => `${BASE}/lab/videos/${videoId}/file`

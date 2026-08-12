// 크롭 랩 산출물 — 목록 · 진행률 · 다운로드 · 이름변경 · 삭제.
//
// 진행 상태는 서버가 progress.jsonl 에서 만들어 준다. 브라우저는 무엇도 기억하지
// 않는다 — 새로고침하든 다른 기기에서 열든 이 목록이 곧 현황이다.
import { BASE, api } from './http'

export type CropRunStatus = 'running' | 'done' | 'error' | 'cancelled'

/** crop.json 의 커버리지 요약 — 외부 계약 스키마라 키가 camelCase 다. */
export interface CropSummary {
  ballTrackingCoverage?: number
  playerTrackingCoverage?: number
  fallbackCoverage?: number
  keyframeCount?: number
}

export interface CropRunOut {
  id: string
  name: string
  kind: 'json' | 'cut' // json = 좌표만 | cut = 추론 없는 크롭 컷
  source_name: string
  created_at: string
  status: CropRunStatus
  error: string | null
  settings: Record<string, unknown>
  summary: CropSummary | null
  has_crop_json: boolean
  has_video: boolean
  video_expired: boolean // 영상만 TTL 로 지워졌다 — 항목 자체는 남는다
}

export interface CropProgressEvent {
  phase: 'start' | 'crop_analyze' | 'annotate' | 'encoding' | 'done' | 'error' | 'cancelled'
  done?: number
  total?: number
  msg?: string
}

export const listCropRuns = (projectId: string) =>
  api.get<CropRunOut[]>(`/projects/${projectId}/crops`)

export const getCropRun = (projectId: string, cropId: string) =>
  api.get<CropRunOut>(`/projects/${projectId}/crops/${cropId}`)

export const renameCropRun = (projectId: string, cropId: string, name: string) =>
  api.patch<CropRunOut>(`/projects/${projectId}/crops/${cropId}`, { name })

export const deleteCropRun = (projectId: string, cropId: string) =>
  api.delete<void>(`/projects/${projectId}/crops/${cropId}`)

export const cancelCropRun = (projectId: string, cropId: string) =>
  api.post(`/projects/${projectId}/crops/${cropId}/cancel`)

export const cropJsonUrl = (projectId: string, cropId: string) =>
  `${BASE}/projects/${projectId}/crops/${cropId}/crop.json`

export const cropVideoUrl = (projectId: string, cropId: string) =>
  `${BASE}/projects/${projectId}/crops/${cropId}/video`

export function subscribeCropEvents(
  projectId: string,
  cropId: string,
  onEvent: (ev: CropProgressEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}/projects/${projectId}/crops/${cropId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as CropProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') source.close()
  })
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}

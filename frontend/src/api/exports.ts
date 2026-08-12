// 데이터셋 내보내기 — 생성 · 진행률 · 목록 · 다운로드 · 취소.
import { BASE, api } from './http'


export interface ExportCreate {
  kind: 'yolo' | 'images'
  val_split?: number
  seed?: number
  names?: string[] | null
}

export interface ExportOut {
  id: string
  name: string
  kind: 'yolo' | 'images'
  created_at: string
  val_split: number
  seed: number
  train: number
  val: number
  count: number
  classes: number
  size_bytes: number
}

// export now runs as a background job; POST returns the id + running status and
// per-image progress streams via subscribeExportEvents. The finished export
// shows up in listExports once its job reaches 'done'.
export const createExport = (projectId: string, body: ExportCreate) =>
  api.post<{ export_id: string; status: string }>(`/projects/${projectId}/exports`, body)

export interface ExportProgressEvent {
  phase: 'start' | 'copy' | 'zip' | 'done' | 'error' | 'cancelled'
  total?: number
  copied?: number
  count?: number
  train?: number
  val?: number
  msg?: string
}

export function subscribeExportEvents(
  projectId: string,
  exportId: string,
  onEvent: (ev: ExportProgressEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}/projects/${projectId}/exports/${exportId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as ExportProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') source.close()
  })
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}


export const cancelExport = (projectId: string, exportId: string) =>
  api.post(`/projects/${projectId}/exports/${exportId}/cancel`)

export const listExports = (projectId: string) =>
  api.get<ExportOut[]>(`/projects/${projectId}/exports`)

export const renameExport = (projectId: string, exportId: string, name: string) =>
  api.patch<ExportOut>(`/projects/${projectId}/exports/${exportId}`, { name })

export const deleteExport = (projectId: string, exportId: string) =>
  api.delete<void>(`/projects/${projectId}/exports/${exportId}`)

export const exportDownloadUrl = (projectId: string, exportId: string) =>
  `${BASE}/projects/${projectId}/exports/${exportId}/download`

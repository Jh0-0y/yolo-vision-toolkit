// 영상 업로드와 프레임 추출.
import { BASE, api, xhrUpload } from './http'
import type { UploadHandlers } from './http'


export interface VideoUploadParams {
  target_fps: number
  max_frames: number
  start_sec: number
  end_sec: number | null
  dedup: boolean
  dedup_threshold: number
  // 타일링 — 프레임을 학습용 타일로 쪼개 저장
  tile?: boolean
  tile_size?: number
  stride?: number
}

export interface VideoProgressEvent {
  phase: 'start' | 'extract' | 'done' | 'error' | 'cancelled'
  saved?: number
  scanned?: number
  total_frames?: number
  skipped_dup?: number
  src_fps?: number
  step?: number
  msg?: string
}

export function uploadVideo(
  projectId: string,
  file: File,
  params: VideoUploadParams,
  handlers: UploadHandlers = {},
) {
  const form = new FormData()
  form.append('file', file)
  form.append('target_fps', String(params.target_fps))
  form.append('max_frames', String(params.max_frames))
  form.append('start_sec', String(params.start_sec))
  if (params.end_sec != null) form.append('end_sec', String(params.end_sec))
  form.append('dedup', String(params.dedup))
  form.append('dedup_threshold', String(params.dedup_threshold))
  if (params.tile) {
    form.append('tile', 'true')
    if (params.tile_size != null) form.append('tile_size', String(params.tile_size))
    if (params.stride != null) form.append('stride', String(params.stride))
  }
  return xhrUpload<{ video_id: string; filename: string; status: string }>(
    `/projects/${projectId}/videos`,
    form,
    handlers,
  )
}

export function subscribeVideoEvents(
  projectId: string,
  videoId: string,
  onEvent: (ev: VideoProgressEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}/projects/${projectId}/videos/${videoId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as VideoProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') {
      source.close()
    }
  })
  source.onerror = () => {
    // EventSource auto-retries transient drops (readyState CONNECTING); only a
    // terminal failure (e.g. 404 — task gone after a reload) reaches CLOSED.
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}


export const cancelVideo = (projectId: string, videoId: string) =>
  api.post(`/projects/${projectId}/videos/${videoId}/cancel`)

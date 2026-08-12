// 오토라벨링 잡 — 제출 · 진행률 구독 · 취소.
import { BASE, api } from './http'


export interface JobCreate {
  model_ids: string[]
  conf: number
  iou_wbf: number
  imgsz: number
  batch_size: number
  names: string[] | null
  /** class name -> max boxes per image (unspecified classes are uncapped) */
  max_boxes_per_class?: Record<string, number>
}

export interface JobOut {
  id: string
  project_id: string
  status: string
  config: Record<string, unknown>
  result: {
    total: number
    labeled: number
    boxes: number
  } | null
  error: string | null
  created_at: string
}

export interface JobProgressEvent {
  phase: 'inference' | 'done' | 'error' | 'cancelled'
  done?: number
  total?: number
  labeled?: number
  boxes?: number
  msg?: string
}

export function subscribeJobEvents(
  jobId: string,
  onEvent: (ev: JobProgressEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${BASE}/jobs/${jobId}/events`)
  source.addEventListener('progress', (e) => {
    const ev = JSON.parse((e as MessageEvent).data) as JobProgressEvent
    onEvent(ev)
    if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') {
      source.close()
    }
  })
  source.onerror = () => {
    // transient drops auto-reconnect (CONNECTING); a terminal failure (job gone
    // after a reload) reaches CLOSED
    if (source.readyState === EventSource.CLOSED) onError?.()
  }
  return () => source.close()
}


export const cancelAutoLabel = (jobId: string) => api.post(`/jobs/${jobId}/cancel`)

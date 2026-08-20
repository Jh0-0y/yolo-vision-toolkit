import { create } from 'zustand'
import {
  cancelAutoLabel,
  cancelImport,
  cancelTiling,
  importVideo,
  subscribeImportEvents,
  subscribeJobEvents,
  subscribeTilingEvents,
  type ImportProgressEvent,
  type JobProgressEvent,
  type TileProgressEvent,
  type VideoImportParams,
} from '../api/client'

// One global home for every long-running operation, so progress stays visible
// across page navigation (the XHR / SSE live here, not in a page component) and
// server jobs reconnect after a full reload.
//
// Two shapes:
//  - `import` — 브라우저가 영상을 올리는 단계(업로드) + 서버가 프레임을 뽑는 단계.
//    앱 안에서 이동해도 살아 있지만, 새로고침하면 업로드는 끊긴다.
//  - `autolabel` — 순수 SSE. 서버가 progress.jsonl 을 처음부터 재생하므로 새로고침도
//    견딘다 — 잡 참조를 저장해 두고 hydrate 때 다시 붙는다.
//  - `tiling` — 순수 SSE. 검수완료 이미지를 타일로 쪼개 **새 데이터셋**을 만든다.
//    결과를 볼 자리가 따로 있지만 사용자는 원본 화면에 그대로 머무르므로,
//    진행률은 여기서 들고 완료 카드가 파생 데이터셋으로 가는 링크를 준다.
//
// 크롭 런과 학습 런은 **여기 없다.** 각자 상세페이지가 진행률을 직접 보여준다 —
// progress.jsonl 을 처음부터 재생하므로 전역 카드에 얹지 않아도 견딘다.
export type JobKind = 'import' | 'autolabel' | 'tiling'
export type JobStatus = 'running' | 'done' | 'error'

export interface JobPhase {
  key: string
  label: string
  indeterminate: boolean
  value: number // 0-100
  detail?: string
}

export interface Job {
  id: string
  kind: JobKind
  title: string
  projectId?: string
  status: JobStatus
  phaseIndex: number
  phases: JobPhase[]
  error?: string
  seq: number // bumps once on done, so consumers react exactly once
  datasetId?: string // import jobs: which dataset the frames land in
  refId?: string // server-side id: import jobId / autolabel jobId
  // The card IS the handle to this job's result, so it waits for the user
  // instead of auto-closing after a few seconds.
  sticky?: boolean
  resultHref?: string // in-app route to the finished result
  resultLabel?: string
}

interface JobStore {
  jobs: Record<string, Job>
  order: string[]
  startDatasetImport: (
    projectId: string,
    datasetId: string,
    file: File,
    params: VideoImportParams,
  ) => string
  trackAutoLabel: (
    projectId: string,
    datasetId: string,
    jobId: string,
    title: string,
  ) => string
  trackTiling: (
    projectId: string,
    datasetId: string,
    tileDatasetId: string,
    title: string,
  ) => string
  cancel: (id: string) => void
  dismiss: (id: string) => void
  hydrate: () => void
}

// abort handles for in-flight client uploads (not part of serializable state)
const aborters = new Map<string, () => void>()

// ---- id generation (Math.random is fine in the browser) ----
let seqCounter = 0
const newId = () => `job_${Date.now().toString(36)}_${(seqCounter++).toString(36)}`

// ---- localStorage: only reconnectable (pure server) jobs are persisted ----
const LS_KEY = 'yvt.active.jobs'
interface PersistedJob {
  id: string
  kind: JobKind
  title: string
  projectId: string
  datasetId: string
  refId: string
}
function readPersisted(): PersistedJob[] {
  try {
    const raw = localStorage.getItem(LS_KEY)
    return raw ? (JSON.parse(raw) as PersistedJob[]) : []
  } catch {
    return []
  }
}
function writePersisted(list: PersistedJob[]): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(list))
  } catch {
    // best-effort — quota / disabled storage
  }
}
function persistAdd(v: PersistedJob): void {
  writePersisted([...readPersisted().filter((x) => x.id !== v.id), v])
}
function persistRemove(id: string): void {
  writePersisted(readPersisted().filter((x) => x.id !== id))
}

// ---- event → phase mappers (one per server-job kind) ----
type Mapped = { phase: JobPhase; status: JobStatus; error?: string }

function mapExtract(ev: ImportProgressEvent): Mapped {
  const pct =
    ev.total_frames && ev.scanned
      ? Math.min(100, Math.round((ev.scanned / ev.total_frames) * 100))
      : ev.phase === 'done'
        ? 100
        : 0
  const phase: JobPhase = {
    key: 'extract',
    label: 'Extracting frames',
    indeterminate: ev.phase === 'start',
    value: pct,
    detail:
      ev.phase === 'start'
        ? `Analyzing… (${ev.total_frames ?? '?'} frames)`
        : ev.phase === 'done'
          ? `${ev.saved ?? 0} frames extracted`
          : `${ev.saved ?? 0} saved (scanned ${ev.scanned ?? 0}/${ev.total_frames ?? '?'})`,
  }
  if (ev.phase === 'done') return { phase, status: 'done' }
  if (ev.phase === 'error' || ev.phase === 'cancelled')
    return { phase, status: 'error', error: ev.msg || 'Extraction stopped' }
  return { phase, status: 'running' }
}

function mapAutolabel(ev: JobProgressEvent): Mapped {
  const pct =
    ev.total && ev.done !== undefined
      ? Math.min(100, Math.round((ev.done / ev.total) * 100))
      : ev.phase === 'done'
        ? 100
        : 0
  const phase: JobPhase = {
    key: 'labeling',
    label: 'Auto-labeling',
    indeterminate: !ev.total,
    value: pct,
    detail:
      ev.phase === 'done'
        ? `${ev.labeled ?? '-'} images labeled`
        : `${ev.done ?? 0}/${ev.total ?? '?'}`,
  }
  if (ev.phase === 'done') return { phase, status: 'done' }
  if (ev.phase === 'error' || ev.phase === 'cancelled')
    return { phase, status: 'error', error: ev.msg || 'Job stopped' }
  return { phase, status: 'running' }
}

function mapTiling(ev: TileProgressEvent): Mapped {
  const pct =
    ev.total && ev.done !== undefined
      ? Math.min(100, Math.round((ev.done / ev.total) * 100))
      : ev.phase === 'done'
        ? 100
        : 0
  const phase: JobPhase = {
    key: 'tiling',
    label: 'Tiling',
    indeterminate: !ev.total,
    value: pct,
    detail:
      ev.phase === 'done'
        ? `${ev.saved ?? ev.tiles ?? '-'} tiles`
        : `${ev.done ?? 0}/${ev.total ?? '?'}`,
  }
  if (ev.phase === 'done') return { phase, status: 'done' }
  if (ev.phase === 'error' || ev.phase === 'cancelled')
    return { phase, status: 'error', error: ev.msg || 'Tiling stopped' }
  return { phase, status: 'running' }
}

export const useJobStore = create<JobStore>((set, get) => {
  const patch = (id: string, fn: (j: Job) => Job) =>
    set((s) => {
      const j = s.jobs[id]
      if (!j) return s
      return { jobs: { ...s.jobs, [id]: fn(j) } }
    })

  const addJob = (job: Job) =>
    set((s) => ({ jobs: { ...s.jobs, [job.id]: job }, order: [...s.order, job.id] }))

  // drive a single-phase server job from its SSE, with reconnect-failure handling
  const attachServerJob = <E>(
    id: string,
    subscribe: (onEvent: (ev: E) => void, onError: () => void) => () => void,
    map: (ev: E) => Mapped,
  ) => {
    subscribe(
      (ev) => {
        const { phase, status, error } = map(ev)
        if (status === 'done') {
          patch(id, (j) => ({ ...j, status: 'done', phaseIndex: 0, phases: [phase], seq: j.seq + 1 }))
          persistRemove(id)
        } else if (status === 'error') {
          patch(id, (j) => ({ ...j, status: 'error', error }))
          persistRemove(id)
        } else {
          patch(id, (j) => ({ ...j, phaseIndex: 0, phases: [phase] }))
        }
      },
      () => {
        patch(id, (j) => ({ ...j, status: 'error', error: 'Lost connection to job' }))
        persistRemove(id)
      },
    )
  }

  const serverPhase = (label: string): JobPhase => ({
    key: 'run',
    label,
    indeterminate: true,
    value: 0,
    detail: 'Reconnecting…',
  })

  return {
    jobs: {},
    order: [],

    // 데이터셋으로 영상 가져오기. 두 단계(업로드 → 추출)이고, 프레임이 데이터셋
    // 안으로 들어간 뒤 **영상은 서버가 지운다.**
    startDatasetImport: (projectId, datasetId, file, params) => {
      const id = newId()
      const controller = new AbortController()
      aborters.set(id, () => controller.abort())
      addJob({
        id,
        kind: 'import',
        title: file.name,
        projectId,
        datasetId,
        status: 'running',
        phaseIndex: 0,
        phases: [
          { key: 'upload', label: 'Uploading video', indeterminate: false, value: 0 },
          { key: 'extract', label: 'Extracting frames', indeterminate: true, value: 0 },
        ],
        seq: 0,
      })
      importVideo(projectId, datasetId, file, params, {
        signal: controller.signal,
        onProgress: (pct) =>
          patch(id, (j) => ({
            ...j,
            phaseIndex: 0,
            phases: [{ ...j.phases[0], value: pct }, j.phases[1]],
          })),
        onUploaded: () =>
          patch(id, (j) => ({
            ...j,
            phaseIndex: 1,
            phases: [{ ...j.phases[0], value: 100 }, j.phases[1]],
          })),
      })
        .then(({ job_id }) => {
          patch(id, (j) => ({ ...j, phaseIndex: 1, refId: job_id }))
          subscribeImportEvents(projectId, datasetId, job_id, (ev) => {
            const { phase, status, error } = mapExtract(ev)
            if (status === 'done') {
              patch(id, (j) => ({
                ...j,
                status: 'done',
                phaseIndex: 1,
                phases: [j.phases[0], phase],
                seq: j.seq + 1,
              }))
            } else if (status === 'error') {
              patch(id, (j) => ({ ...j, status: 'error', error }))
            } else {
              patch(id, (j) => ({ ...j, phaseIndex: 1, phases: [j.phases[0], phase] }))
            }
          })
        })
        .catch((e) => patch(id, (j) => ({ ...j, status: 'error', error: String(e) })))
      return id
    },

    trackAutoLabel: (projectId, datasetId, jobId, title) => {
      const id = newId()
      addJob({
        id,
        kind: 'autolabel',
        title,
        projectId,
        datasetId,
        status: 'running',
        phaseIndex: 0,
        phases: [{ key: 'labeling', label: 'Auto-labeling', indeterminate: true, value: 0 }],
        seq: 0,
        refId: jobId,
      })
      persistAdd({ id, kind: 'autolabel', title, projectId, datasetId, refId: jobId })
      attachServerJob(id, (onEv, onErr) => subscribeJobEvents(jobId, onEv, onErr), mapAutolabel)
      return id
    },

    trackTiling: (projectId, datasetId, tileDatasetId, title) => {
      const id = newId()
      addJob({
        id,
        kind: 'tiling',
        title,
        projectId,
        datasetId,
        status: 'running',
        phaseIndex: 0,
        phases: [{ key: 'tiling', label: 'Tiling', indeterminate: true, value: 0 }],
        seq: 0,
        refId: tileDatasetId,
        // 카드가 결과로 가는 유일한 손잡이라 자동으로 닫지 않는다
        sticky: true,
        resultHref: `/projects/${projectId}/datasets/${tileDatasetId}`,
        resultLabel: 'Open dataset',
      })
      persistAdd({ id, kind: 'tiling', title, projectId, datasetId, refId: tileDatasetId })
      attachServerJob(
        id,
        (onEv, onErr) => subscribeTilingEvents(projectId, datasetId, tileDatasetId, onEv, onErr),
        mapTiling,
      )
      return id
    },

    cancel: (id) => {
      const j = get().jobs[id]
      if (!j || j.status !== 'running') return
      aborters.get(id)?.() // abort an in-flight client upload (no-op if finished)
      if (j.refId && j.projectId) {
        if (j.kind === 'import' && j.datasetId)
          cancelImport(j.projectId, j.datasetId, j.refId).catch(() => {})
        else if (j.kind === 'autolabel') cancelAutoLabel(j.refId).catch(() => {})
        else if (j.kind === 'tiling' && j.datasetId)
          cancelTiling(j.projectId, j.datasetId, j.refId).catch(() => {})
      }
      // reflect immediately; the SSE / upload rejection will also settle it
      patch(id, (jj) => ({ ...jj, status: 'error', error: 'Cancelled' }))
      persistRemove(id)
    },

    dismiss: (id) =>
      set((s) => {
        aborters.delete(id)
        const jobs = { ...s.jobs }
        delete jobs[id]
        persistRemove(id)
        return { jobs, order: s.order.filter((x) => x !== id) }
      }),

    // 저장되는 것은 **순수 서버 잡**뿐이다 — 지금은 오토라벨링과 타일링.
    // 가져오기는 브라우저가 바이트를 쥐고 있어 새로고침하면 어차피 끊긴다.
    hydrate: () => {
      for (const v of readPersisted()) {
        if (get().jobs[v.id]) continue
        if (v.kind === 'autolabel') {
          addJob({
            id: v.id,
            kind: v.kind,
            title: v.title,
            projectId: v.projectId,
            datasetId: v.datasetId,
            status: 'running',
            phaseIndex: 0,
            phases: [serverPhase('Auto-labeling')],
            seq: 0,
            refId: v.refId,
          })
          attachServerJob(v.id, (onEv, onErr) => subscribeJobEvents(v.refId, onEv, onErr), mapAutolabel)
          continue
        }
        if (v.kind === 'tiling') {
          addJob({
            id: v.id,
            kind: 'tiling',
            title: v.title,
            projectId: v.projectId,
            datasetId: v.datasetId,
            status: 'running',
            phaseIndex: 0,
            phases: [serverPhase('Tiling')],
            seq: 0,
            refId: v.refId,
            sticky: true,
            resultHref: `/projects/${v.projectId}/datasets/${v.refId}`,
            resultLabel: 'Open dataset',
          })
          attachServerJob(
            v.id,
            (onEv, onErr) => subscribeTilingEvents(v.projectId, v.datasetId, v.refId, onEv, onErr),
            mapTiling,
          )
          continue
        }
        persistRemove(v.id)
      }
    },
  }
})

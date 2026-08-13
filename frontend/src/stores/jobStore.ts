import { create } from 'zustand'
import {
  cancelAutoLabel,
  cancelCropRun,
  cancelExport,
  cancelLiveJob,
  cancelVideo,
  listCropRuns,
  subscribeCropEvents,
  subscribeExportEvents,
  subscribeJobEvents,
  subscribeLiveEvents,
  subscribeVideoEvents,
  uploadTrainDataset,
  uploadVideo,
  type CropProgressEvent,
  type ExportProgressEvent,
  type JobProgressEvent,
  type LiveProgress,
  type VideoProgressEvent,
  type VideoUploadParams,
} from '../api/client'

// One global home for every long-running operation, so progress stays visible
// across page navigation (the XHR / SSE live here, not in a page component) and
// server jobs reconnect after a full reload.
//
// Three shapes:
//  - client uploads (dataset zip, video file): a browser-driven upload phase,
//    then a server phase. Survives in-app nav; a full reload aborts the upload.
//  - server jobs (video-extract after upload, auto-label, export): pure SSE with
//    a progress.jsonl the server replays from the start, so they survive a full
//    reload too — we persist the job ref and re-subscribe on hydrate.
//  - crop runs: the same SSE, but the server ALSO lists what is running
//    (GET /crops derives status from progress.jsonl), so nothing is persisted
//    here — `syncCropRuns` asks the server instead. That survives a different
//    browser, a different machine, and a cleared localStorage.
//  - live (Crop Draw detect / overlay render): the same SSE. Not persisted here
//    either — the Draw tab remembers its session id and re-registers the job on
//    mount if the server says it is still running.
export type JobKind = 'dataset' | 'video' | 'autolabel' | 'export' | 'crop' | 'live'
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
  refId?: string // server-side id: videoId / autolabel jobId / exportId / cropId
  resultToken?: string // dataset jobs: the created "upload:{id}" token
  // The card IS the handle to this job's result, so it waits for the user
  // instead of auto-closing after a few seconds.
  sticky?: boolean
  resultHref?: string // in-app route to the finished result
  resultLabel?: string
}

interface JobStore {
  jobs: Record<string, Job>
  order: string[]
  startDatasetUpload: (file: File) => string
  startVideoJob: (projectId: string, file: File, params: VideoUploadParams) => string
  trackAutoLabel: (projectId: string, jobId: string, title: string) => string
  trackExport: (projectId: string, exportId: string, title: string) => string
  trackCrop: (projectId: string, cropId: string, title: string) => string
  trackLive: (projectId: string, jobId: string, title: string) => string
  cancel: (id: string) => void
  dismiss: (id: string) => void
  hydrate: () => void
  syncCropRuns: (projectId: string) => Promise<void>
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

function mapVideo(ev: VideoProgressEvent): Mapped {
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
        ? `Analyzing… (source ${ev.src_fps ?? '?'}fps, ${ev.total_frames ?? '?'} frames)`
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

function mapExport(ev: ExportProgressEvent): Mapped {
  let phase: JobPhase
  if (ev.phase === 'copy') {
    const pct = ev.total ? Math.min(100, Math.round(((ev.copied ?? 0) / ev.total) * 100)) : 0
    phase = { key: 'export', label: 'Exporting', indeterminate: false, value: pct, detail: `${ev.copied ?? 0}/${ev.total ?? '?'} images` }
  } else if (ev.phase === 'zip') {
    phase = { key: 'export', label: 'Exporting', indeterminate: true, value: 100, detail: 'Zipping…' }
  } else if (ev.phase === 'done') {
    phase = { key: 'export', label: 'Exporting', indeterminate: false, value: 100, detail: `${ev.count ?? ev.train ?? 0} images` }
  } else if (ev.phase === 'error') {
    phase = { key: 'export', label: 'Exporting', indeterminate: false, value: 0, detail: ev.msg }
  } else {
    phase = { key: 'export', label: 'Exporting', indeterminate: true, value: 0, detail: 'Preparing…' }
  }
  if (ev.phase === 'done') return { phase, status: 'done' }
  if (ev.phase === 'error' || ev.phase === 'cancelled')
    return { phase, status: 'error', error: ev.msg || 'Cancelled' }
  return { phase, status: 'running' }
}

const CROP_PHASE_DETAIL: Record<string, string> = {
  start: 'Preparing…',
  crop_analyze: 'Analyzing crop trajectory…',
  annotate: 'Rendering video…',
  encoding: 'Encoding video…',
}

function mapCrop(ev: CropProgressEvent): Mapped {
  const pct =
    ev.total && ev.done != null
      ? Math.min(100, Math.round((ev.done / ev.total) * 100))
      : ev.phase === 'done'
        ? 100
        : 0
  const phase: JobPhase = {
    key: 'crop',
    label: 'Crop',
    // only the render pass reports done/total; the rest have no measurable length
    indeterminate: ev.phase !== 'annotate' && ev.phase !== 'done',
    value: pct,
    detail: ev.phase === 'done' ? 'Saved' : (CROP_PHASE_DETAIL[ev.phase] ?? 'Working…'),
  }
  if (ev.phase === 'done') return { phase, status: 'done' }
  if (ev.phase === 'error' || ev.phase === 'cancelled')
    return { phase, status: 'error', error: ev.msg || 'Crop job stopped' }
  return { phase, status: 'running' }
}

const LIVE_PHASE_DETAIL: Record<string, string> = {
  start: 'Preparing…',
  detect: 'Detecting objects…',
  render: 'Rendering overlay…',
  encoding: 'Encoding video…',
}

function mapLive(ev: LiveProgress): Mapped {
  const pct =
    ev.total && ev.done != null
      ? Math.min(100, Math.round((ev.done / ev.total) * 100))
      : ev.phase === 'done'
        ? 100
        : 0
  const phase: JobPhase = {
    key: 'live',
    label: 'Crop Draw',
    // only the detect / render passes count frames; the rest have no measurable length
    indeterminate: ev.phase !== 'detect' && ev.phase !== 'render' && ev.phase !== 'done',
    value: pct,
    detail: ev.phase === 'done' ? 'Ready' : (LIVE_PHASE_DETAIL[ev.phase] ?? 'Working…'),
  }
  if (ev.phase === 'done') return { phase, status: 'done' }
  if (ev.phase === 'error' || ev.phase === 'cancelled')
    return { phase, status: 'error', error: ev.msg || 'Crop Draw job stopped' }
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

    startDatasetUpload: (file) => {
      const id = newId()
      const controller = new AbortController()
      aborters.set(id, () => controller.abort())
      addJob({
        id,
        kind: 'dataset',
        title: file.name,
        status: 'running',
        phaseIndex: 0,
        phases: [
          { key: 'upload', label: 'Uploading', indeterminate: false, value: 0 },
          { key: 'process', label: 'Processing', indeterminate: true, value: 0 },
        ],
        seq: 0,
      })
      uploadTrainDataset(file, {
        signal: controller.signal,
        onProgress: (pct) =>
          patch(id, (j) => ({ ...j, phaseIndex: 0, phases: [{ ...j.phases[0], value: pct }, j.phases[1]] })),
        onUploaded: () => patch(id, (j) => ({ ...j, phaseIndex: 1 })),
      })
        .then((d) =>
          patch(id, (j) => ({
            ...j,
            status: 'done',
            phaseIndex: 1,
            phases: [{ ...j.phases[0], value: 100 }, { ...j.phases[1], indeterminate: false, value: 100 }],
            seq: j.seq + 1,
            resultToken: d.dataset,
          })),
        )
        .catch((e) => patch(id, (j) => ({ ...j, status: 'error', error: String(e) })))
      return id
    },

    startVideoJob: (projectId, file, params) => {
      const id = newId()
      const controller = new AbortController()
      aborters.set(id, () => controller.abort())
      addJob({
        id,
        kind: 'video',
        title: file.name,
        projectId,
        status: 'running',
        phaseIndex: 0,
        phases: [
          { key: 'upload', label: 'Uploading video', indeterminate: false, value: 0 },
          { key: 'extract', label: 'Extracting frames', indeterminate: true, value: 0 },
        ],
        seq: 0,
      })
      uploadVideo(projectId, file, params, {
        signal: controller.signal,
        onProgress: (pct) =>
          patch(id, (j) => ({ ...j, phaseIndex: 0, phases: [{ ...j.phases[0], value: pct }, j.phases[1]] })),
        onUploaded: () =>
          patch(id, (j) => ({ ...j, phaseIndex: 1, phases: [{ ...j.phases[0], value: 100 }, j.phases[1]] })),
      })
        .then(({ video_id }) => {
          patch(id, (j) => ({ ...j, phaseIndex: 1, refId: video_id }))
          persistAdd({ id, kind: 'video', title: file.name, projectId, refId: video_id })
          // video's second phase reuses the extract mapper; wrap to prefill phase[0]=100
          subscribeVideoEvents(
            projectId,
            video_id,
            (ev) => {
              const { phase, status, error } = mapVideo(ev)
              if (status === 'done') {
                patch(id, (j) => ({ ...j, status: 'done', phaseIndex: 1, phases: [j.phases[0], phase], seq: j.seq + 1 }))
                persistRemove(id)
              } else if (status === 'error') {
                patch(id, (j) => ({ ...j, status: 'error', error }))
                persistRemove(id)
              } else {
                patch(id, (j) => ({ ...j, phaseIndex: 1, phases: [j.phases[0], phase] }))
              }
            },
            () => {
              patch(id, (j) => ({ ...j, status: 'error', error: 'Lost connection to extraction' }))
              persistRemove(id)
            },
          )
        })
        .catch((e) => patch(id, (j) => ({ ...j, status: 'error', error: String(e) })))
      return id
    },

    trackAutoLabel: (projectId, jobId, title) => {
      const id = newId()
      addJob({
        id,
        kind: 'autolabel',
        title,
        projectId,
        status: 'running',
        phaseIndex: 0,
        phases: [{ key: 'labeling', label: 'Auto-labeling', indeterminate: true, value: 0 }],
        seq: 0,
        refId: jobId,
      })
      persistAdd({ id, kind: 'autolabel', title, projectId, refId: jobId })
      attachServerJob(id, (onEv, onErr) => subscribeJobEvents(jobId, onEv, onErr), mapAutolabel)
      return id
    },

    trackExport: (projectId, exportId, title) => {
      const id = newId()
      addJob({
        id,
        kind: 'export',
        title,
        projectId,
        status: 'running',
        phaseIndex: 0,
        phases: [{ key: 'export', label: 'Exporting', indeterminate: true, value: 0 }],
        seq: 0,
        refId: exportId,
      })
      persistAdd({ id, kind: 'export', title, projectId, refId: exportId })
      attachServerJob(id, (onEv, onErr) => subscribeExportEvents(projectId, exportId, onEv, onErr), mapExport)
      return id
    },

    trackCrop: (projectId, cropId, title) => {
      const id = newId()
      addJob({
        id,
        kind: 'crop',
        title,
        projectId,
        status: 'running',
        phaseIndex: 0,
        phases: [{ key: 'crop', label: 'Crop', indeterminate: true, value: 0, detail: 'Preparing…' }],
        seq: 0,
        refId: cropId,
        // nothing in localStorage: syncCropRuns re-finds this from the server
        sticky: true,
        resultHref: `/projects/${projectId}/lab/crops`,
        resultLabel: 'Open Crop Runs',
      })
      attachServerJob(id, (onEv, onErr) => subscribeCropEvents(projectId, cropId, onEv, onErr), mapCrop)
      return id
    },

    trackLive: (projectId, jobId, title) => {
      const id = newId()
      addJob({
        id,
        kind: 'live',
        title,
        projectId,
        status: 'running',
        phaseIndex: 0,
        phases: [{ key: 'live', label: 'Crop Draw', indeterminate: true, value: 0, detail: 'Preparing…' }],
        seq: 0,
        refId: jobId,
        // 결과를 볼 곳이 Draw 탭뿐이라 카드가 그 손잡이다
        sticky: true,
        resultHref: `/projects/${projectId}/lab/crop?tab=draw`,
        resultLabel: 'Open Crop Draw',
      })
      attachServerJob(id, (onEv, onErr) => subscribeLiveEvents(jobId, onEv, onErr), mapLive)
      return id
    },

    syncCropRuns: async (projectId) => {
      let runs
      try {
        runs = await listCropRuns(projectId)
      } catch {
        return // offline / project gone — the Crop Runs page reports it properly
      }
      const { jobs, order, trackCrop } = get()
      const tracked = new Set(
        order.map((i) => jobs[i]).filter((j) => j?.kind === 'crop').map((j) => j.refId),
      )
      for (const run of runs) {
        if (run.status !== 'running' || tracked.has(run.id)) continue
        trackCrop(projectId, run.id, run.name)
      }
    },

    cancel: (id) => {
      const j = get().jobs[id]
      if (!j || j.status !== 'running') return
      aborters.get(id)?.() // abort an in-flight client upload (no-op if finished)
      if (j.refId && j.projectId) {
        if (j.kind === 'video') cancelVideo(j.projectId, j.refId).catch(() => {})
        else if (j.kind === 'autolabel') cancelAutoLabel(j.refId).catch(() => {})
        else if (j.kind === 'export') cancelExport(j.projectId, j.refId).catch(() => {})
        else if (j.kind === 'crop') cancelCropRun(j.projectId, j.refId).catch(() => {})
        else if (j.kind === 'live') cancelLiveJob(j.refId).catch(() => {})
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

    hydrate: () => {
      for (const v of readPersisted()) {
        if (get().jobs[v.id]) continue
        const label =
          v.kind === 'autolabel' ? 'Auto-labeling' : v.kind === 'export' ? 'Exporting' : 'Extracting frames'
        addJob({
          id: v.id,
          kind: v.kind,
          title: v.title,
          projectId: v.projectId,
          status: 'running',
          phaseIndex: 0,
          phases: [serverPhase(label)],
          seq: 0,
          refId: v.refId,
        })
        if (v.kind === 'video') {
          attachServerJob(v.id, (onEv, onErr) => subscribeVideoEvents(v.projectId, v.refId, onEv, onErr), mapVideo)
        } else if (v.kind === 'autolabel') {
          attachServerJob(v.id, (onEv, onErr) => subscribeJobEvents(v.refId, onEv, onErr), mapAutolabel)
        } else if (v.kind === 'export') {
          attachServerJob(v.id, (onEv, onErr) => subscribeExportEvents(v.projectId, v.refId, onEv, onErr), mapExport)
        }
      }
    },
  }
})

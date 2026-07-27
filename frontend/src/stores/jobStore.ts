import { create } from 'zustand'
import {
  subscribeVideoEvents,
  uploadTrainDataset,
  uploadVideo,
  type VideoProgressEvent,
  type VideoUploadParams,
} from '../api/client'

// A background job with ordered phases, tracked globally so its progress
// survives in-app navigation (the XHR / SSE live here, not in a page component).
//
// Two kinds, two persistence guarantees:
//  - dataset zip:  [upload → process]  — client-driven upload; survives in-app
//    navigation only (a full reload aborts the byte transfer).
//  - video:        [upload → extract]  — the extract phase is a server job with
//    an SSE stream, so it survives even a full reload: we persist the video_id
//    and re-subscribe on app load (hydrate).
export type JobKind = 'dataset' | 'video'
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
  seq: number // bumps once when the job reaches done, so consumers react once
  videoId?: string // set once the video upload returns; used to reconnect SSE
  resultToken?: string // dataset jobs: the created "upload:{id}" token
}

interface JobStore {
  jobs: Record<string, Job>
  order: string[] // insertion order for stable rendering
  startDatasetUpload: (file: File) => string
  startVideoJob: (projectId: string, file: File, params: VideoUploadParams) => string
  dismiss: (id: string) => void
  hydrate: () => void // reconnect persisted video jobs on app load
}

// ---- id generation (Math.random is fine in the browser) ----
let seqCounter = 0
const newId = () => `job_${Date.now().toString(36)}_${(seqCounter++).toString(36)}`

// ---- localStorage: only reconnectable (video/extract) jobs are persisted ----
const LS_KEY = 'yvt.active.videos'
interface PersistedVideo {
  id: string
  title: string
  projectId: string
  videoId: string
}
function readPersisted(): PersistedVideo[] {
  try {
    const raw = localStorage.getItem(LS_KEY)
    return raw ? (JSON.parse(raw) as PersistedVideo[]) : []
  } catch {
    return []
  }
}
function writePersisted(list: PersistedVideo[]): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(list))
  } catch {
    // ignore quota / disabled storage — persistence is best-effort
  }
}
function persistAdd(v: PersistedVideo): void {
  const list = readPersisted().filter((x) => x.id !== v.id)
  list.push(v)
  writePersisted(list)
}
function persistRemove(id: string): void {
  writePersisted(readPersisted().filter((x) => x.id !== id))
}

// map a video SSE event onto the extract phase (index 1)
function extractPhaseFromEvent(ev: VideoProgressEvent): JobPhase {
  if (ev.phase === 'start') {
    return {
      key: 'extract',
      label: 'Extracting frames',
      indeterminate: true,
      value: 0,
      detail: `Analyzing… (source ${ev.src_fps ?? '?'}fps, ${ev.total_frames ?? '?'} frames)`,
    }
  }
  const pct =
    ev.total_frames && ev.scanned
      ? Math.min(100, Math.round((ev.scanned / ev.total_frames) * 100))
      : ev.phase === 'done'
        ? 100
        : 0
  return {
    key: 'extract',
    label: 'Extracting frames',
    indeterminate: false,
    value: pct,
    detail:
      ev.phase === 'done'
        ? `${ev.saved ?? 0} frames extracted`
        : `${ev.saved ?? 0} saved (scanned ${ev.scanned ?? 0}/${ev.total_frames ?? '?'})`,
  }
}

export const useJobStore = create<JobStore>((set, get) => {
  // patch a single job in place
  const patch = (id: string, fn: (j: Job) => Job) =>
    set((s) => {
      const j = s.jobs[id]
      if (!j) return s
      return { jobs: { ...s.jobs, [id]: fn(j) } }
    })

  // subscribe to a video's server-side extraction stream and drive its phase
  const attachVideoStream = (id: string, projectId: string, videoId: string) => {
    subscribeVideoEvents(
      projectId,
      videoId,
      (ev) => {
        const phase = extractPhaseFromEvent(ev)
        if (ev.phase === 'done') {
          patch(id, (j) => ({
            ...j,
            status: 'done',
            phaseIndex: 1,
            phases: [j.phases[0], phase],
            seq: j.seq + 1,
          }))
          persistRemove(id)
        } else if (ev.phase === 'error' || ev.phase === 'cancelled') {
          patch(id, (j) => ({
            ...j,
            status: 'error',
            error: ev.msg || 'Extraction stopped',
          }))
          persistRemove(id)
        } else {
          patch(id, (j) => ({ ...j, phaseIndex: 1, phases: [j.phases[0], phase] }))
        }
      },
      () => {
        // stream failed to (re)connect — the task is gone; drop the job
        patch(id, (j) => ({ ...j, status: 'error', error: 'Lost connection to extraction' }))
        persistRemove(id)
      },
    )
  }

  const addJob = (job: Job) =>
    set((s) => ({ jobs: { ...s.jobs, [job.id]: job }, order: [...s.order, job.id] }))

  return {
    jobs: {},
    order: [],

    startDatasetUpload: (file) => {
      const id = newId()
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
        onProgress: (pct) =>
          patch(id, (j) => ({
            ...j,
            phaseIndex: 0,
            phases: [{ ...j.phases[0], value: pct }, j.phases[1]],
          })),
        onUploaded: () => patch(id, (j) => ({ ...j, phaseIndex: 1 })),
      })
        .then((d) =>
          patch(id, (j) => ({
            ...j,
            status: 'done',
            phaseIndex: 1,
            phases: [{ ...j.phases[0], value: 100 }, { ...j.phases[1], indeterminate: false, value: 100 }],
            seq: j.seq + 1,
            // the created token, for the Train page to auto-select on completion
            resultToken: d.dataset,
          })),
        )
        .catch((e) => patch(id, (j) => ({ ...j, status: 'error', error: String(e) })))
      return id
    },

    startVideoJob: (projectId, file, params) => {
      const id = newId()
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
        .then(({ video_id }) => {
          patch(id, (j) => ({ ...j, phaseIndex: 1, videoId: video_id }))
          persistAdd({ id, title: file.name, projectId, videoId: video_id })
          attachVideoStream(id, projectId, video_id)
        })
        .catch((e) => patch(id, (j) => ({ ...j, status: 'error', error: String(e) })))
      return id
    },

    dismiss: (id) =>
      set((s) => {
        const jobs = { ...s.jobs }
        delete jobs[id]
        persistRemove(id)
        return { jobs, order: s.order.filter((x) => x !== id) }
      }),

    hydrate: () => {
      for (const v of readPersisted()) {
        if (get().jobs[v.id]) continue // already tracked this session
        addJob({
          id: v.id,
          kind: 'video',
          title: v.title,
          projectId: v.projectId,
          status: 'running',
          phaseIndex: 1,
          phases: [
            { key: 'upload', label: 'Uploading video', indeterminate: false, value: 100 },
            { key: 'extract', label: 'Extracting frames', indeterminate: true, value: 0, detail: 'Reconnecting…' },
          ],
          seq: 0,
          videoId: v.videoId,
        })
        attachVideoStream(v.id, v.projectId, v.videoId)
      }
    },
  }
})

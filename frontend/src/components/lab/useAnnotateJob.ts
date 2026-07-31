import { useEffect, useRef, useState } from 'react'
import {
  startAnnotate,
  subscribeAnnotateEvents,
  type AnnotateProgress,
} from '../../api/client'

// Inference image size — usually matched to the model's training size (multiples of 32).
export const IMGSZ_OPTIONS = ['640', '960', '1280', '1600', '1920']
export const DEFAULT_IMGSZ = 1280

export const PHASE_LABEL: Record<string, string> = {
  start: 'Preparing…',
  crop_analyze: 'Analyzing crop trajectory…',
  annotate: 'Rendering video…',
  encoding: 'Encoding video…',
  done: 'Done',
}

type StartOpts = Parameters<typeof startAnnotate>[0]

/** Shared hook for the annotate job lifecycle (drop → start → SSE progress → done/error/cancel).
 *  Reused by both Lab pages (Crop Result / Crop Draw Tool). */
export function useAnnotateJob() {
  const [fileName, setFileName] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<AnnotateProgress | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const unsub = useRef<(() => void) | null>(null)

  useEffect(() => () => unsub.current?.(), [])

  // Generic launcher — takes a display name and a starter that returns { job_id }.
  async function launch(name: string, starter: () => Promise<{ job_id: string }>) {
    unsub.current?.()
    setFileName(name)
    setError(null)
    setDone(false)
    setJobId(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      const { job_id } = await starter()
      setJobId(job_id)
      unsub.current = subscribeAnnotateEvents(job_id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          setDone(true)
          setRunning(false)
        } else if (ev.phase === 'error') {
          setError(ev.msg || 'Tracking failed')
          setRunning(false)
        } else if (ev.phase === 'cancelled') {
          setRunning(false)
        }
      })
    } catch (e) {
      setError((e as Error).message)
      setRunning(false)
    }
  }

  const run = (opts: StartOpts) => launch(opts.file.name, () => startAnnotate(opts))

  function reset() {
    unsub.current?.()
    setFileName(null)
    setJobId(null)
    setProgress(null)
    setRunning(false)
    setError(null)
    setDone(false)
  }

  const pct =
    progress?.total && progress.done != null
      ? Math.round((progress.done / progress.total) * 100)
      : 0

  return { fileName, jobId, progress, running, error, done, pct, setError, run, launch, reset }
}

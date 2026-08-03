import { useEffect, useRef, useState } from 'react'
import { startLive, subscribeLiveEvents, type LiveProgress } from '../../api/client'

export const LIVE_PHASE_LABEL: Record<string, string> = {
  start: 'Preparing…',
  detect: 'Detecting objects…',
  encoding: 'Encoding preview…',
  done: 'Done',
}

type StartOpts = Parameters<typeof startLive>[0]

/** Lifecycle hook for the live-preview detection pass (drop → detect → done).
 *  On success it calls `onDone(detectId)` so the caller can fetch the cached result. */
export function useLiveJob(onDone: (detectId: string) => void) {
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<LiveProgress | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const unsub = useRef<(() => void) | null>(null)
  const onDoneRef = useRef(onDone)
  onDoneRef.current = onDone

  useEffect(() => () => unsub.current?.(), [])

  async function run(opts: StartOpts) {
    unsub.current?.()
    setError(null)
    setJobId(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      const { job_id } = await startLive(opts)
      setJobId(job_id)
      unsub.current = subscribeLiveEvents(job_id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          setRunning(false)
          onDoneRef.current(job_id)
        } else if (ev.phase === 'error') {
          setError(ev.msg || 'Detection failed')
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

  function reset() {
    unsub.current?.()
    setJobId(null)
    setProgress(null)
    setRunning(false)
    setError(null)
  }

  const pct =
    progress?.total && progress.done != null
      ? Math.round((progress.done / progress.total) * 100)
      : 0

  return { jobId, progress, running, error, pct, setError, run, reset }
}

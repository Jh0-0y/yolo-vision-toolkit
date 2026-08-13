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

  function listen(id: string) {
    unsub.current?.()
    setError(null)
    setJobId(id)
    setProgress({ phase: 'start' })
    setRunning(true)
    unsub.current = subscribeLiveEvents(id, (ev) => {
      setProgress(ev)
      if (ev.phase === 'done') {
        setRunning(false)
        onDoneRef.current(id)
      } else if (ev.phase === 'error') {
        setError(ev.msg || 'Detection failed')
        setRunning(false)
      } else if (ev.phase === 'cancelled') {
        setRunning(false)
      }
    })
  }

  async function run(opts: StartOpts) {
    unsub.current?.()
    setError(null)
    setJobId(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      listen((await startLive(opts)).job_id)
    } catch (e) {
      setError((e as Error).message)
      setRunning(false)
    }
  }

  /** 이미 돌고 있는 검출에 다시 붙는다 (탭 복귀·새로고침).
   *  서버가 progress.jsonl 을 처음부터 재생하므로 늦게 붙어도 같은 그림이 나온다. */
  const attach = (id: string) => listen(id)

  function reset() {
    unsub.current?.()
    unsub.current = null
    setJobId(null)
    setProgress(null)
    setRunning(false)
    setError(null)
  }

  const pct =
    progress?.total && progress.done != null
      ? Math.round((progress.done / progress.total) * 100)
      : 0

  return { jobId, progress, running, error, pct, setError, run, attach, reset }
}

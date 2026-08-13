import { useState } from 'react'
import { startLive, startLiveRender, type TrackcropOverrides } from '../../api/client'
import { useJobStore } from '../../stores/jobStore'

type StartOpts = Parameters<typeof startLive>[0]

/** Crop Draw 의 서버 잡을 **시작**하고 전역 잡 카드에 넘긴다.
 *
 * 검출 패스도 오버레이 렌더도 몇 분씩 걸리는 서버 잡이라, 다른 오래 걸리는 작업과
 * 같은 자리(우상단 카드)에서 보여야 한다. 이 화면은 진행률을 그리지 않고 "끝났다"는
 * 신호만 받아 결과를 가져온다.
 *
 * 카드가 `refId` 로 SSE 를 듣고 취소도 보내므로, 여기서는 잡 하나의 스토어 id 만
 * 들고 있으면 된다. */
export function useLiveJob(projectId: string) {
  const trackLive = useJobStore((s) => s.trackLive)
  const [jobKey, setJobKey] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const job = useJobStore((s) => (jobKey ? s.jobs[jobKey] : undefined))

  async function launch(title: string, starter: () => Promise<{ job_id: string }>) {
    setError(null)
    setStarting(true)
    try {
      const { job_id } = await starter()
      setJobKey(trackLive(projectId, job_id, title))
      return job_id
    } catch (e) {
      setError((e as Error).message)
      return null
    } finally {
      setStarting(false)
    }
  }

  const run = (opts: StartOpts) => launch(opts.file.name, () => startLive(opts))

  const render = (detectId: string, title: string, overrides: TrackcropOverrides, toggles: Record<string, boolean>) =>
    launch(title, () => startLiveRender(detectId, overrides, toggles))

  /** 이미 돌고 있는 잡을 카드에 올린다 (탭 복귀·새로고침).
   *  서버가 progress.jsonl 을 처음부터 재생하므로 늦게 붙어도 같은 그림이 나온다. */
  const attach = (jobId: string, title: string) => setJobKey(trackLive(projectId, jobId, title))

  function reset() {
    setJobKey(null)
    setError(null)
  }

  return {
    starting,
    error,
    setError,
    /** 마지막으로 시작·복구한 잡. 진행률 UI 는 전역 카드가 그린다 — 여기서는 상태만 본다. */
    status: job?.status,
    /** done 으로 넘어갈 때 한 번 올라간다 */
    seq: job?.seq,
    jobError: job?.error,
    run,
    render,
    attach,
    reset,
  }
}

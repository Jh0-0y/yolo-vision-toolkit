import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { startAnnotate } from '../../api/client'
import { useJobStore } from '../../stores/jobStore'

// Inference image size — usually matched to the model's training size (multiples of 32).
export const IMGSZ_OPTIONS = ['640', '960', '1280', '1600', '1920']
export const DEFAULT_IMGSZ = 1280

type StartOpts = Parameters<typeof startAnnotate>[0]

/** 크롭 잡을 **시작**만 하는 훅.
 *
 * 진행률과 결과는 이 화면이 들고 있지 않다 — 시작하자마자 `jobStore` 에 넘기면
 * 전역 카드가 진행률을 보여주고, 산출물은 서버의 크롭 런 목록에 남는다. 그래서
 * 탭을 옮기든 새로고침하든 잡이 사라지지 않는다. */
export function useAnnotateJob(projectId: string) {
  const trackCrop = useJobStore((s) => s.trackCrop)
  const queryClient = useQueryClient()
  const [starting, setStarting] = useState(false)
  const [startedName, setStartedName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Generic launcher — takes a display name and a starter that returns { job_id }.
  async function launch(name: string, starter: () => Promise<{ job_id: string }>) {
    setError(null)
    setStarting(true)
    try {
      const { job_id } = await starter()
      trackCrop(projectId, job_id, name)
      setStartedName(name)
      queryClient.invalidateQueries({ queryKey: ['crop-runs', projectId] })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setStarting(false)
    }
  }

  const run = (opts: StartOpts) => launch(opts.file.name, () => startAnnotate(opts))

  return { starting, startedName, error, setError, run, launch }
}

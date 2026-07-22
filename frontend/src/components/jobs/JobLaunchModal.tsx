import { useEffect, useRef, useState } from 'react'
import {
  Button,
  Group,
  Modal,
  MultiSelect,
  NumberInput,
  Progress,
  Select,
  Stack,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  subscribeJobEvents,
  type JobOut,
  type JobProgressEvent,
  type ModelOut,
} from '../../api/client'

interface Props {
  projectId: string
  opened: boolean
  onClose: () => void
}

export default function JobLaunchModal({ projectId, opened, onClose }: Props) {
  const queryClient = useQueryClient()
  const [modelIds, setModelIds] = useState<string[]>([])
  const [confConfirm, setConfConfirm] = useState<number | string>(0.6)
  const [confMin, setConfMin] = useState<number | string>(0.1)
  const [iouWbf, setIouWbf] = useState<number | string>(0.55)
  const [emptyPolicy, setEmptyPolicy] = useState<string | null>('review')
  const [progress, setProgress] = useState<JobProgressEvent | null>(null)
  const [runningJobId, setRunningJobId] = useState<string | null>(null)
  const unsubscribe = useRef<(() => void) | null>(null)

  const models = useQuery({
    queryKey: ['models'],
    queryFn: () => api.get<ModelOut[]>('/models'),
  })

  useEffect(() => () => unsubscribe.current?.(), [])

  const launch = useMutation({
    mutationFn: () =>
      api.post<JobOut>(`/projects/${projectId}/jobs`, {
        model_ids: modelIds,
        conf_confirm: Number(confConfirm),
        conf_min: Number(confMin),
        iou_wbf: Number(iouWbf),
        empty_policy: emptyPolicy ?? 'review',
      }),
    onSuccess: (job) => {
      setRunningJobId(job.id)
      setProgress({ phase: 'inference', done: 0 })
      unsubscribe.current = subscribeJobEvents(job.id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          notifications.show({
            message: `라벨링 완료 — 확정 ${ev.confirmed ?? '-'}, 리뷰 ${ev.review ?? '-'}`,
            color: 'green',
          })
          finish()
        } else if (ev.phase === 'error') {
          notifications.show({ message: `잡 실패: ${ev.msg ?? '알 수 없는 오류'}`, color: 'red' })
          finish()
        } else if (ev.phase === 'cancelled') {
          notifications.show({ message: '잡이 취소되었습니다', color: 'yellow' })
          finish()
        }
      })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const finish = () => {
    queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    queryClient.invalidateQueries({ queryKey: ['images', projectId] })
    queryClient.invalidateQueries({ queryKey: ['jobs', projectId] })
    setRunningJobId(null)
  }

  const cancel = () => {
    if (runningJobId) api.post(`/jobs/${runningJobId}/cancel`)
  }

  const running = runningJobId !== null
  const pct =
    progress?.total && progress.done !== undefined
      ? Math.round((progress.done / progress.total) * 100)
      : 0

  return (
    <Modal opened={opened} onClose={onClose} title="오토라벨링 실행" size="lg" closeOnClickOutside={!running}>
      <Stack>
        <MultiSelect
          label="모델 (1~N개, 클래스가 자동으로 union됩니다)"
          data={models.data?.map((m) => ({
            value: m.id,
            label: `${m.name} (${Object.keys(m.classes).length}개 클래스)`,
          })) ?? []}
          value={modelIds}
          onChange={setModelIds}
          disabled={running}
        />
        <Group grow>
          <NumberInput
            label="자동 확정 임계값"
            description="이 신뢰도 이상이면 자동 확정"
            value={confConfirm}
            onChange={setConfConfirm}
            min={0.1}
            max={1}
            step={0.05}
            decimalScale={2}
            disabled={running}
          />
          <NumberInput
            label="최소 신뢰도"
            description="이 미만 박스는 무시"
            value={confMin}
            onChange={setConfMin}
            min={0.01}
            max={0.9}
            step={0.05}
            decimalScale={2}
            disabled={running}
          />
        </Group>
        <Group grow>
          <NumberInput
            label="WBF IoU"
            description="복수 모델 박스 병합 기준"
            value={iouWbf}
            onChange={setIouWbf}
            min={0.3}
            max={0.9}
            step={0.05}
            decimalScale={2}
            disabled={running}
          />
          <Select
            label="빈 이미지 처리"
            data={[
              { value: 'review', label: '리뷰로 보내기' },
              { value: 'negative', label: '네거티브로 확정 (빈 라벨)' },
            ]}
            value={emptyPolicy}
            onChange={setEmptyPolicy}
            disabled={running}
          />
        </Group>

        {running && progress && (
          <Stack gap="xs">
            <Progress value={pct} animated />
            <Text size="sm" c="dimmed">
              {progress.done ?? 0}/{progress.total ?? '?'} — 확정 {progress.confirmed ?? 0} · 리뷰{' '}
              {progress.review ?? 0} · 네거티브 {progress.negative ?? 0}
            </Text>
          </Stack>
        )}

        <Group justify="flex-end">
          {running ? (
            <Button color="red" variant="light" onClick={cancel}>
              취소
            </Button>
          ) : (
            <Button onClick={() => launch.mutate()} disabled={modelIds.length === 0} loading={launch.isPending}>
              실행
            </Button>
          )}
        </Group>
      </Stack>
    </Modal>
  )
}

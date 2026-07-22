import { useEffect, useRef, useState } from 'react'
import {
  Button,
  Collapse,
  Group,
  Modal,
  MultiSelect,
  NumberInput,
  Progress,
  Stack,
  Text,
} from '@mantine/core'
import { IconSettings } from '@tabler/icons-react'
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
  /** target file names; null = every image in the project */
  names: string[] | null
}

export default function AutoLabelModal({ projectId, opened, onClose, names }: Props) {
  const queryClient = useQueryClient()
  const [modelIds, setModelIds] = useState<string[]>([])
  const [conf, setConf] = useState<number | string>(0.4)
  const [iouWbf, setIouWbf] = useState<number | string>(0.55)
  const [imgsz, setImgsz] = useState<number | string>(640)
  const [advanced, setAdvanced] = useState(false)
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
        conf: Number(conf),
        iou_wbf: Number(iouWbf),
        imgsz: Number(imgsz),
        names,
      }),
    onSuccess: (job) => {
      setRunningJobId(job.id)
      setProgress({ phase: 'inference', done: 0 })
      unsubscribe.current = subscribeJobEvents(job.id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          notifications.show({
            message: `Auto labeling done — ${ev.labeled ?? ev.done ?? '-'} images labeled`,
            color: 'green',
          })
          finish()
        } else if (ev.phase === 'error') {
          notifications.show({ message: `Job failed: ${ev.msg ?? 'unknown error'}`, color: 'red' })
          finish()
        } else if (ev.phase === 'cancelled') {
          notifications.show({ message: 'Job cancelled', color: 'yellow' })
          finish()
        }
      })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const finish = () => {
    queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    queryClient.invalidateQueries({ queryKey: ['images', projectId] })
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
    <Modal
      opened={opened}
      onClose={onClose}
      title="Auto Labeling"
      size="lg"
      closeOnClickOutside={!running}
    >
      <Stack>
        <Text size="sm" c="dimmed">
          Target: {names ? `${names.length} selected images` : 'all images in the project'} —
          results will be marked as needing review.
        </Text>

        <MultiSelect
          label="Models (1–N, classes are unioned by name)"
          data={
            models.data?.map((m) => ({
              value: m.id,
              label: `${m.name} (${Object.keys(m.classes).length} classes)`,
            })) ?? []
          }
          value={modelIds}
          onChange={setModelIds}
          disabled={running}
        />
        <NumberInput
          label="Confidence threshold (conf)"
          description="Only boxes above this confidence are kept"
          value={conf}
          onChange={setConf}
          min={0.05}
          max={0.95}
          step={0.05}
          decimalScale={2}
          disabled={running}
        />

        <Button
          variant="subtle"
          size="compact-sm"
          leftSection={<IconSettings size={14} />}
          onClick={() => setAdvanced((v) => !v)}
          style={{ alignSelf: 'flex-start' }}
        >
          Advanced
        </Button>
        <Collapse expanded={advanced}>
          <Group grow>
            <NumberInput
              label="WBF IoU"
              description="IoU for merging boxes across models"
              value={iouWbf}
              onChange={setIouWbf}
              min={0.3}
              max={0.9}
              step={0.05}
              decimalScale={2}
              disabled={running}
            />
            <NumberInput
              label="Image size (imgsz)"
              value={imgsz}
              onChange={setImgsz}
              min={320}
              max={1920}
              step={32}
              disabled={running}
            />
          </Group>
        </Collapse>

        {running && progress && (
          <Stack gap="xs">
            <Progress value={pct} animated />
            <Text size="sm" c="dimmed">
              {progress.done ?? 0}/{progress.total ?? '?'} — {progress.boxes ?? 0} boxes
            </Text>
          </Stack>
        )}

        <Group justify="flex-end">
          {running ? (
            <Button color="red" variant="light" onClick={cancel}>
              Cancel
            </Button>
          ) : (
            <Button
              onClick={() => launch.mutate()}
              disabled={modelIds.length === 0}
              loading={launch.isPending}
            >
              Run
            </Button>
          )}
        </Group>
      </Stack>
    </Modal>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
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
  const [maxBoxes, setMaxBoxes] = useState<Record<string, number>>({})
  const [advanced, setAdvanced] = useState(false)
  const [progress, setProgress] = useState<JobProgressEvent | null>(null)
  const [runningJobId, setRunningJobId] = useState<string | null>(null)
  const unsubscribe = useRef<(() => void) | null>(null)

  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })

  // Union of class names across the selected models (classes merge by name).
  const classNames = useMemo(() => {
    const selected = new Set(modelIds)
    const names = new Set<string>()
    for (const m of models.data ?? []) {
      if (selected.has(m.id)) {
        for (const n of Object.values(m.classes)) names.add(n)
      }
    }
    return [...names].sort((a, b) => a.localeCompare(b))
  }, [models.data, modelIds])

  const DEFAULT_MAX = 300

  useEffect(() => () => unsubscribe.current?.(), [])

  const launch = useMutation({
    mutationFn: () =>
      api.post<JobOut>(`/projects/${projectId}/jobs`, {
        model_ids: modelIds,
        conf: Number(conf),
        iou_wbf: Number(iouWbf),
        imgsz: Number(imgsz),
        names,
        // only send classes the user capped below the default
        max_boxes_per_class: Object.fromEntries(
          Object.entries(maxBoxes).filter(([, v]) => v !== DEFAULT_MAX),
        ),
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
          Target: {names ? `${names.length} selected images` : 'all images in the project'}.
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
          description="Boxes below this confidence are discarded — not saved to the label file"
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
          <Stack>
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

            {classNames.length > 0 && (
              <Stack gap="xs">
                <Text size="sm" fw={500}>
                  Max boxes per class
                </Text>
                <Text size="xs" c="dimmed">
                  Keep only the N highest-confidence boxes of each class per image (e.g. ball = 1).
                  Leave at {DEFAULT_MAX} for no limit.
                </Text>
                {classNames.map((name) => (
                  <NumberInput
                    key={name}
                    label={name}
                    value={maxBoxes[name] ?? DEFAULT_MAX}
                    onChange={(v) =>
                      setMaxBoxes((prev) => ({ ...prev, [name]: Number(v) || 0 }))
                    }
                    min={0}
                    step={1}
                    disabled={running}
                  />
                ))}
              </Stack>
            )}
          </Stack>
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

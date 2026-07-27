import { useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Collapse,
  Group,
  Modal,
  MultiSelect,
  NumberInput,
  Stack,
  Text,
} from '@mantine/core'
import { IconAlertTriangle, IconSettings } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, getResources, type JobOut, type ModelOut } from '../../api/client'
import { useJobStore } from '../../stores/jobStore'

interface Props {
  projectId: string
  opened: boolean
  onClose: () => void
  /** target file names; null = every image in the project */
  names: string[] | null
}

export default function AutoLabelModal({ projectId, opened, onClose, names }: Props) {
  const startAutoLabel = useJobStore((s) => s.trackAutoLabel)
  const [modelIds, setModelIds] = useState<string[]>([])
  const [conf, setConf] = useState<number | string>(0.4)
  const [iouWbf, setIouWbf] = useState<number | string>(0.55)
  const [imgsz, setImgsz] = useState<number | string>(640)
  const [maxBoxes, setMaxBoxes] = useState<Record<string, number>>({})
  const [advanced, setAdvanced] = useState(false)

  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })

  // warn about GPU contention while a training run is active
  const resources = useQuery({
    queryKey: ['resources'],
    queryFn: getResources,
    refetchInterval: opened ? 5000 : false,
    enabled: opened,
  })
  const trainingActive = resources.data?.training_active ?? false

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
      // hand off to the global job indicator — progress survives closing this
      // modal, navigating away, and a full reload (SSE reconnect).
      startAutoLabel(
        projectId,
        job.id,
        names ? `Auto-label · ${names.length} imgs` : 'Auto-label · all',
      )
      onClose()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const running = launch.isPending

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Auto Labeling"
      size="lg"
      closeOnClickOutside={!running}
    >
      <Stack>
        {trainingActive && (
          <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="학습 진행 중">
            학습이 진행 중입니다. 오토라벨링은 같은 GPU를 사용하므로 느려지거나 메모리 부족이 날 수
            있습니다.
          </Alert>
        )}
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

        <Group justify="flex-end">
          <Button
            onClick={() => launch.mutate()}
            disabled={modelIds.length === 0}
            loading={launch.isPending}
          >
            Run
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}

import { useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Collapse,
  Group,
  Modal,
  NumberInput,
  Stack,
  Text,
} from '@mantine/core'
import { IconAlertTriangle, IconSettings } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, getResources, type JobOut, type ModelOut } from '../../api/client'
import DetectorList, { newEntry, type DetectorEntry } from '../detect/DetectorList'
import { useJobStore } from '../../stores/jobStore'

interface Props {
  projectId: string
  /** 라벨이 들어갈 데이터셋 — 이미지도 클래스도 그 안에 있다 */
  datasetId: string
  opened: boolean
  onClose: () => void
  /** 대상 이미지 파일명. **비울 수 없다** — 오토라벨링은 고른 것에만 돈다. */
  names: string[]
}

export default function AutoLabelModal({
  projectId,
  datasetId,
  opened,
  onClose,
  names,
}: Props) {
  const startAutoLabel = useJobStore((s) => s.trackAutoLabel)
  // 엔트리마다 모델과 **추론 방식**을 따로 갖는다 — 선수는 풀 프레임, 공은 타일 식으로.
  const [entries, setEntries] = useState<DetectorEntry[]>(() => [newEntry('full')])
  const [conf, setConf] = useState<number | string>(0.4)
  const [iouWbf, setIouWbf] = useState<number | string>(0.55)
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
    const selected = new Set(entries.map((e) => e.modelId).filter(Boolean) as string[])
    const names = new Set<string>()
    for (const m of models.data ?? []) {
      if (selected.has(m.id)) {
        for (const n of Object.values(m.classes)) names.add(n)
      }
    }
    return [...names].sort((a, b) => a.localeCompare(b))
  }, [models.data, entries])

  const DEFAULT_MAX = 300

  const launch = useMutation({
    mutationFn: () =>
      api.post<JobOut>(`/projects/${projectId}/jobs`, {
        dataset_id: datasetId,
        detectors: entries
          .filter((e) => e.modelId)
          .map((e) => ({
            model_id: e.modelId,
            mode: e.mode,
            imgsz: e.imgsz,
            tile_size: e.tileSize,
            stride: e.stride,
            merge_iou: e.mergeIou,
            border_margin_px: e.borderMargin ?? 4,
          })),
        conf: Number(conf),
        iou_wbf: Number(iouWbf),
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
        datasetId,
        job.id,
        `Auto-label · ${names.length} imgs`,
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
          <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="Training in progress">
            A training run is active. Auto-labeling uses the same GPU, so both may slow down or run
            out of memory.
          </Alert>
        )}
        <Text size="sm" c="dimmed">
          Target: {names.length} selected images.
        </Text>

        <DetectorList
          models={models.data ?? []}
          entries={entries}
          onEntries={setEntries}
          disabled={running}
          showConf={false}
          showBorderMargin
          defaultMode="full"
        />
        <NumberInput
          label="Confidence threshold (conf)"
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
                value={iouWbf}
                onChange={setIouWbf}
                min={0.3}
                max={0.9}
                step={0.05}
                decimalScale={2}
                disabled={running}
              />
            </Group>

            {classNames.length > 0 && (
              <Stack gap="xs">
                <Text size="sm" fw={500}>
                  Max boxes per class
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
            disabled={!entries.some((e) => e.modelId)}
            loading={launch.isPending}
          >
            Run
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}

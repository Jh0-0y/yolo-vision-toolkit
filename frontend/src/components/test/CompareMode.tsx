// 벤치마크 목록 — 런 하나가 디렉터리 하나이고 이력이 남는다.
// 결과는 상세 페이지에서 본다(학습이 이력→상세로 나뉜 것과 같은 결).

import { useLayoutEffect, useState } from 'react'
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Modal,
  Select,
  Slider,
  Stack,
  Table,
  Text,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconAlertTriangle, IconChartBar, IconPlus, IconTrash } from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import type { ModelOut } from '../../api/client'
import { listDatasets } from '../../api/datasets'
import {
  deleteBenchmark,
  listBenchmarks,
  startBenchmark,
  type BenchmarkEntry,
} from '../../api/test/compare'
import DetectorList, { newEntry, type DetectorEntry } from '../detect/DetectorList'

const STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  cancelled: 'yellow',
}

interface Props {
  projectId: string
  models: ModelOut[]
  /** 새 벤치마크 모달의 열림 상태. 여는 버튼이 ModelsPage 헤더(Registry 의 Add model 자리)에
   *  올라가 있어, 상태도 부모가 들고 있다. */
  newBenchmarkOpened: boolean
  onNewBenchmarkOpened: (opened: boolean) => void
}

/** 화면의 `DetectorEntry` → API 의 `BenchmarkEntry`. 엔트리별 conf 는 없다 —
 *  모든 엔트리가 모달의 전역 conf 하나로 채점된다. */
const toApiEntry = (e: DetectorEntry): BenchmarkEntry => ({
  model_id: e.modelId!,
  mode: e.mode,
  imgsz: e.imgsz,
  tile_size: e.tileSize,
  stride: e.stride,
  merge_iou: e.mergeIou,
  border_margin_px: e.borderMargin ?? 4,
})

/** 벤치마크 이력 + 새로 만들기. 박스 오버레이·지표 표는 상세 페이지(Task 7)로 옮겼다. */
export default function CompareMode({
  projectId,
  models,
  newBenchmarkOpened,
  onNewBenchmarkOpened,
}: Props) {
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [datasetToken, setDatasetToken] = useState<string | null>(null)
  const [conf, setConf] = useState(0.4)
  const [iou, setIou] = useState(0.5)
  const [entries, setEntries] = useState<DetectorEntry[]>([newEntry('full')])

  const datasets = useQuery({
    queryKey: ['datasets', projectId],
    queryFn: () => listDatasets(projectId),
    enabled: newBenchmarkOpened,
  })

  const benchmarks = useQuery({
    queryKey: ['benchmarks', projectId],
    queryFn: () => listBenchmarks(projectId),
    // 도는 런이 있으면 상태가 바뀌니 짧게 당겨 폴링한다
    refetchInterval: 3_000,
  })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['benchmarks', projectId] })

  const remove = useMutation({
    mutationFn: (id: string) => deleteBenchmark(projectId, id),
    onSuccess: () => {
      notifications.show({ message: 'Benchmark deleted', color: 'green' })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const start = useMutation({
    mutationFn: () =>
      startBenchmark({
        dataset: datasetToken!,
        entries: entries.map(toApiEntry),
        conf,
        iou,
      }),
    onSuccess: ({ job_id }) => {
      onNewBenchmarkOpened(false)
      invalidate()
      navigate(`/projects/${projectId}/benchmarks/${job_id}`)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  // 모달이 **열리는 순간** 입력을 기본값으로 되돌린다. 여는 버튼이 둘(헤더 · 빈 상태)이고
  // 하나는 부모에 있어, 준비는 여는 쪽이 아니라 열림 상태에 붙어 있어야 빠지지 않는다.
  // 그리기 전에 끝나야 지난 런의 값이 한 프레임 비치지 않으므로 layout effect 다.
  useLayoutEffect(() => {
    if (!newBenchmarkOpened) return
    setDatasetToken(null)
    setConf(0.4)
    setIou(0.5)
    setEntries([newEntry('full')])
  }, [newBenchmarkOpened])

  const openModal = () => onNewBenchmarkOpened(true)

  const canStart = !!datasetToken && entries.length > 0 && !entries.some((e) => !e.modelId)
  const rows = benchmarks.data ?? []

  return (
    <Stack gap="lg">
      <Card withBorder radius="md" padding="sm">
        <Table highlightOnHover verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Dataset</Table.Th>
              <Table.Th>Models</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Created</Table.Th>
              <Table.Th w={48} />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((b) => (
              <Table.Tr
                key={b.id}
                style={{ cursor: 'pointer' }}
                onClick={() => navigate(`/projects/${projectId}/benchmarks/${b.id}`)}
              >
                <Table.Td>
                  <Text size="sm" fw={600}>
                    {b.dataset_name}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{b.entries} models</Text>
                </Table.Td>
                <Table.Td>
                  <Group gap={6} wrap="nowrap">
                    {b.status === 'running' && <Loader size={14} />}
                    <Badge variant="light" color={STATUS_COLOR[b.status] ?? 'gray'}>
                      {b.status}
                    </Badge>
                  </Group>
                  {b.error && (
                    <Text size="xs" c="red" lineClamp={1}>
                      {b.error}
                    </Text>
                  )}
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed">
                    {new Date(b.created_at).toLocaleString()}
                  </Text>
                </Table.Td>
                <Table.Td onClick={(e) => e.stopPropagation()}>
                  <Tooltip label="Delete">
                    <ActionIcon
                      variant="subtle"
                      color="red"
                      loading={remove.isPending && remove.variables === b.id}
                      onClick={() => {
                        if (confirm(`Delete benchmark on "${b.dataset_name}"?`))
                          remove.mutate(b.id)
                      }}
                    >
                      <IconTrash size={16} />
                    </ActionIcon>
                  </Tooltip>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
        {rows.length === 0 && (
          <Stack align="center" gap="xs" py="xl">
            <IconChartBar size={36} stroke={1.2} />
            <Text c="dimmed" size="sm">
              No benchmarks yet. Score models against a dataset&apos;s test split.
            </Text>
            <Button variant="light" leftSection={<IconPlus size={16} />} onClick={openModal}>
              New benchmark
            </Button>
          </Stack>
        )}
      </Card>

      <Modal
        opened={newBenchmarkOpened}
        onClose={() => onNewBenchmarkOpened(false)}
        title="New benchmark"
        size="lg"
      >
        <Stack gap="md">
          {/* 고를 수 있는 것이 하나도 없으면 Select 가 통째로 잠긴 것처럼 보인다 — 이유를 말해 준다 */}
          {(datasets.data ?? []).length > 0 &&
            (datasets.data ?? []).every((d) => d.test === 0) && (
              <Alert color="yellow" icon={<IconAlertTriangle size={16} />}>
                No dataset has a test split yet. Open a dataset, split it into train/val/test
                with a test ratio above 0, and it will show up here.
              </Alert>
            )}

          <Select
            label="Dataset"
            description="Scored against this dataset's test split."
            placeholder={datasets.data?.length ? 'Pick a dataset' : 'No datasets yet'}
            // test 가 비어 있으면 시작해도 422 다 — 누르게 두지 말고 여기서 막고,
            // 왜 못 고르는지 수치로 보여 준다.
            data={(datasets.data ?? []).map((d) => ({
              value: `dataset:${projectId}:${d.id}`,
              label: d.test > 0 ? `${d.name} — ${d.test} in test` : `${d.name} — no test split`,
              disabled: d.test === 0,
            }))}
            value={datasetToken}
            onChange={setDatasetToken}
            disabled={start.isPending}
          />

          <DetectorList
            models={models}
            entries={entries}
            onEntries={setEntries}
            disabled={start.isPending}
            showConf={false}
            showBorderMargin
            defaultMode="full"
          />

          <Group grow align="flex-start">
            <div>
              <Text size="sm" fw={600}>
                Confidence <Text span c="dimmed">{conf.toFixed(2)}</Text>
              </Text>
              <Slider
                min={0.05}
                max={0.95}
                step={0.05}
                value={conf}
                onChange={setConf}
                disabled={start.isPending}
              />
            </div>
            <div>
              <Text size="sm" fw={600}>
                Match IoU <Text span c="dimmed">{iou.toFixed(2)}</Text>
              </Text>
              <Slider
                min={0.3}
                max={0.9}
                step={0.05}
                value={iou}
                onChange={setIou}
                disabled={start.isPending}
              />
            </div>
          </Group>

          <Button onClick={() => start.mutate()} disabled={!canStart} loading={start.isPending}>
            Start
          </Button>
        </Stack>
      </Modal>
    </Stack>
  )
}

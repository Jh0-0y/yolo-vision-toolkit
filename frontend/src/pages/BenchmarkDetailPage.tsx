// 벤치마크 결과 상세 — 엔트리별 지표 카드 · 클래스별 표 · 박스 오버레이.
// 표시 로직은 옛 Compare 탭에서 그대로 옮겨 왔고, 키만 모델에서 **엔트리**로 바꿨다:
// 같은 모델이 방식만 달리해 두 번 들어올 수 있어 `model_id` 로는 구분되지 않는다.

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ActionIcon,
  Alert,
  Badge,
  Card,
  Group,
  Loader,
  Modal,
  Progress,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { BarChart } from '@mantine/charts'
import { IconAlertTriangle, IconArrowLeft } from '@tabler/icons-react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import {
  ApiError,
  getBenchmarkResult,
  listBenchmarks,
  subscribeBenchmarkEvents,
  type CompareBox,
  type CompareEntryResult,
  type CompareImage,
  type CompareProgress,
} from '../api/client'

const GT_COLOR = '#51cf66' // ground truth = green (dashed)
const ENTRY_COLORS = ['#4dabf7', '#f783ac', '#ffa94d', '#845ef7', '#38d9a9', '#ff8787']

const STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  cancelled: 'yellow',
}

interface Layer {
  boxes: CompareBox[]
  color: string
  dashed: boolean
}

function BoxOverlay({ src, layers }: { src: string; layers: Layer[] }) {
  return (
    <div style={{ position: 'relative', width: '100%', lineHeight: 0 }}>
      <img src={src} alt="" style={{ width: '100%', display: 'block', borderRadius: 6 }} />
      {layers.flatMap((layer, li) =>
        layer.boxes.map((b, i) => {
          const [x1, y1, x2, y2] = b.xyxyn
          return (
            <div
              key={`${li}-${i}`}
              style={{
                position: 'absolute',
                left: `${x1 * 100}%`,
                top: `${y1 * 100}%`,
                width: `${(x2 - x1) * 100}%`,
                height: `${(y2 - y1) * 100}%`,
                border: `2px ${layer.dashed ? 'dashed' : 'solid'} ${layer.color}`,
                borderRadius: 2,
              }}
            />
          )
        }),
      )}
    </div>
  )
}

const fmt = (v: number | undefined) => (v == null ? '—' : v.toFixed(3))

/** 데이터셋 test 분할로 채점한 벤치마크 한 건의 결과. 결과가 아직 없으면(404)
 *  진행률을 구독해 기다리고, 오래전에 끝난 런은 구독 없이 바로 결과를 읽는다. */
export default function BenchmarkDetailPage() {
  const { projectId = '', benchId = '' } = useParams()
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [progress, setProgress] = useState<CompareProgress | null>(null)
  const [enlarged, setEnlarged] = useState<CompareImage | null>(null)
  const unsub = useRef<(() => void) | null>(null)
  // 상태가 done 으로 넘어갔는데 결과가 없을 때 한 번만 다시 가져온다 — 조건 없이
  // 다시 부르면 진짜 404 일 때 무한 재조회가 된다.
  const refetchedOnDone = useRef(false)

  // 헤더 정보(데이터셋 이름·엔트리 수·conf/iou·상태)는 목록에서 온다 —
  // 결과가 아직 없어도 헤더는 채워져야 한다.
  const benchmarks = useQuery({
    queryKey: ['benchmarks', projectId],
    queryFn: () => listBenchmarks(projectId),
    // SSE 가 끊겨도 상태 전이를 놓치지 않도록, 도는 동안만 짧게 당겨 본다
    refetchInterval: (q) =>
      q.state.data?.some((b) => b.id === benchId && b.status === 'running') ? 5_000 : false,
  })
  const bench = benchmarks.data?.find((b) => b.id === benchId) ?? null

  const result = useQuery({
    queryKey: ['benchmark', benchId],
    queryFn: () => getBenchmarkResult(projectId, benchId),
    // 404 는 "아직 안 끝났다"는 뜻이다 — 재시도가 아니라 진행률 구독으로 기다린다
    retry: false,
  })
  const data = result.data ?? null
  const { refetch: refetchResult } = result

  const running = bench?.status === 'running'

  // 진행률 구독 — 도는 동안에만. 끝나면 상태가 바뀌며 정리 함수가 스트림을 닫는다.
  useEffect(() => {
    if (!running) return
    unsub.current = subscribeBenchmarkEvents(benchId, (ev) => {
      setProgress(ev)
      if (ev.phase === 'done') qc.invalidateQueries({ queryKey: ['benchmark', benchId] })
      if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled')
        qc.invalidateQueries({ queryKey: ['benchmarks', projectId] })
    })
    return () => {
      unsub.current?.()
      unsub.current = null
    }
  }, [running, benchId, projectId, qc])

  // 이벤트를 한 번도 못 받고 끝난 경우(스트림 유실)의 안전망
  useEffect(() => {
    if (bench?.status === 'done' && !data && !refetchedOnDone.current) {
      refetchedOnDone.current = true
      refetchResult()
    }
  }, [bench?.status, data, refetchResult])

  const pct =
    progress?.total && progress.done != null
      ? Math.round((progress.done / progress.total) * 100)
      : 0

  // 엔트리마다 색과 이름표를 미리 정한다. 이름표는 **모델 이름 + 방식** 이어야
  // 같은 모델의 두 카드가 구분되고, 차트 시리즈 이름으로도 겹치지 않는다.
  const view = useMemo(() => {
    const colors: Record<string, string> = {}
    const labels: Record<string, string> = {}
    const seen = new Map<string, number>()
    ;(data?.per_entry ?? []).forEach((e, i) => {
      colors[e.entry_id] = ENTRY_COLORS[i % ENTRY_COLORS.length]
      const base = `${e.name} · ${e.mode}`
      const n = (seen.get(base) ?? 0) + 1
      seen.set(base, n)
      labels[e.entry_id] = n > 1 ? `${base} #${n}` : base
    })
    return { colors, labels }
  }, [data])

  const colorOf = (entryId: string) => view.colors[entryId] ?? '#868e96'
  const labelOf = (e: CompareEntryResult) => view.labels[e.entry_id] ?? e.name

  // "Best" = highest mAP@0.5:0.95 (the headline COCO metric)
  const bestId = useMemo(() => {
    if (!data?.per_entry.length) return null
    return data.per_entry.reduce((a, b) => (b.map > a.map ? b : a)).entry_id
  }, [data])

  const prfData = useMemo(() => {
    if (!data) return []
    return (['precision', 'recall', 'f1'] as const).map((k) => ({
      metric: k[0].toUpperCase() + k.slice(1),
      ...Object.fromEntries(data.per_entry.map((e) => [view.labels[e.entry_id], e.overall[k]])),
    }))
  }, [data, view])

  const mapData = useMemo(() => {
    if (!data) return []
    return (['map50', 'map'] as const).map((k) => ({
      metric: k === 'map50' ? 'mAP@0.5' : 'mAP@0.5:0.95',
      ...Object.fromEntries(data.per_entry.map((e) => [view.labels[e.entry_id], e[k]])),
    }))
  }, [data, view])

  const series = (data?.per_entry ?? []).map((e) => ({
    name: labelOf(e),
    color: colorOf(e.entry_id),
  }))

  const notReady = result.error instanceof ApiError && result.error.status === 404
  const resultError = result.error && !notReady ? (result.error as Error).message : null
  const conf = bench?.conf ?? data?.conf
  const iou = bench?.iou ?? data?.iou
  const entryCount = data?.per_entry.length ?? bench?.entries
  const waiting = !data && (running || (!bench && benchmarks.isLoading) || result.isFetching)

  return (
    <Stack gap="md">
      {/* header */}
      <Group justify="space-between" wrap="wrap">
        <Group gap="xs">
          <Tooltip label="Back to Benchmarks">
            <ActionIcon variant="default" onClick={() => navigate(`/projects/${projectId}/models`)}>
              <IconArrowLeft size={16} />
            </ActionIcon>
          </Tooltip>
          <Title order={4}>{bench?.dataset_name ?? 'Benchmark'}</Title>
          {bench && (
            <Badge color={STATUS_COLOR[bench.status] ?? 'gray'} variant="light">
              {bench.status}
            </Badge>
          )}
          {running && <Loader size={14} />}
        </Group>
        <Text size="sm" c="dimmed">
          {entryCount != null && `${entryCount} ${entryCount === 1 ? 'entry' : 'entries'}`}
          {conf != null && ` · conf ${conf.toFixed(2)}`}
          {iou != null && ` · match IoU ${iou.toFixed(2)}`}
          {data && ` · ${data.image_count} images`}
        </Text>
      </Group>

      {!benchmarks.isLoading && !bench && (
        <Alert color="gray" icon={<IconAlertTriangle size={18} />}>
          This benchmark is no longer in the history.
        </Alert>
      )}

      {bench?.status === 'error' && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />}>
          {bench.error || progress?.msg || 'Benchmark failed'}
        </Alert>
      )}
      {bench?.status === 'cancelled' && (
        <Alert color="yellow" icon={<IconAlertTriangle size={18} />}>
          This benchmark was cancelled before it finished.
        </Alert>
      )}
      {!data && !waiting && notReady && bench?.status !== 'running' && (
        <Alert color="gray" icon={<IconAlertTriangle size={18} />}>
          No result was stored for this benchmark.
        </Alert>
      )}
      {resultError && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />}>
          {resultError}
        </Alert>
      )}

      {waiting && (
        <Card withBorder radius="md" padding="md">
          <Stack gap={4}>
            <Text size="sm">
              {progress?.phase === 'analyze' || progress?.total
                ? `Scoring… ${progress?.done ?? 0}/${progress?.total ?? '?'} images`
                : 'Waiting for the benchmark to finish…'}
            </Text>
            <Progress value={pct} animated />
          </Stack>
        </Card>
      )}

      {data && (
        <Stack gap="md">
          {data.warning && (
            <Alert color="orange" icon={<IconAlertTriangle size={18} />}>
              {data.warning}
            </Alert>
          )}

          {/* per-entry metrics */}
          <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
            {data.per_entry.map((e) => (
              <Card key={e.entry_id} withBorder radius="md" padding="sm">
                <Group justify="space-between" mb={6}>
                  <Group gap={6}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        background: colorOf(e.entry_id),
                      }}
                    />
                    <Text size="sm" fw={600} truncate="end" maw={160}>
                      {labelOf(e)}
                    </Text>
                  </Group>
                  {e.entry_id === bestId && (
                    <Badge size="xs" color="teal" variant="light">
                      Best mAP
                    </Badge>
                  )}
                </Group>
                <Group grow mb={6}>
                  <Metric label="mAP@.5" value={e.map50.toFixed(3)} strong />
                  <Metric label="mAP@.5:.95" value={e.map.toFixed(3)} strong />
                </Group>
                <Group grow>
                  <Metric label="P" value={e.overall.precision.toFixed(3)} />
                  <Metric label="R" value={e.overall.recall.toFixed(3)} />
                  <Metric label="F1" value={e.overall.f1.toFixed(3)} />
                </Group>
                <Text size="xs" c="dimmed" mt={6}>
                  {e.detections} detections · TP {e.overall.tp} · FP {e.overall.fp} · FN{' '}
                  {e.overall.fn}
                </Text>
              </Card>
            ))}
          </SimpleGrid>

          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
            <Card withBorder radius="md" padding="md">
              <Text size="sm" fw={600} mb="xs">
                mAP by entry
              </Text>
              <BarChart
                h={240}
                data={mapData}
                dataKey="metric"
                series={series}
                yAxisProps={{ domain: [0, 1] }}
                withLegend
              />
            </Card>
            <Card withBorder radius="md" padding="md">
              <Text size="sm" fw={600} mb="xs">
                Precision / Recall / F1 by entry
              </Text>
              <BarChart
                h={240}
                data={prfData}
                dataKey="metric"
                series={series}
                yAxisProps={{ domain: [0, 1] }}
                withLegend
              />
            </Card>
          </SimpleGrid>

          {/* per-class metrics, one table per entry */}
          <Stack gap="xs">
            <Text size="sm" fw={600}>
              Per-class metrics
            </Text>
            <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="sm">
              {data.per_entry.map((e) => (
                <Card key={e.entry_id} withBorder radius="md" padding="sm">
                  <Group gap={6} mb={6}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        background: colorOf(e.entry_id),
                      }}
                    />
                    <Text size="sm" fw={600}>
                      {labelOf(e)}
                    </Text>
                  </Group>
                  <Table.ScrollContainer minWidth={420}>
                    <Table striped highlightOnHover fz="xs" verticalSpacing={4}>
                      <Table.Thead>
                        <Table.Tr>
                          <Table.Th>Class</Table.Th>
                          <Table.Th ta="right">GT</Table.Th>
                          <Table.Th ta="right">P</Table.Th>
                          <Table.Th ta="right">R</Table.Th>
                          <Table.Th ta="right">F1</Table.Th>
                          <Table.Th ta="right">AP@.5</Table.Th>
                          <Table.Th ta="right">AP@.5:.95</Table.Th>
                        </Table.Tr>
                      </Table.Thead>
                      <Table.Tbody>
                        {e.per_class.map((c) => (
                          <Table.Tr key={c.cls}>
                            <Table.Td>{c.name}</Table.Td>
                            <Table.Td ta="right">{c.gt}</Table.Td>
                            <Table.Td ta="right">{fmt(c.precision)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.recall)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.f1)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.ap50)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.ap)}</Table.Td>
                          </Table.Tr>
                        ))}
                        {e.per_class.length === 0 && (
                          <Table.Tr>
                            <Table.Td colSpan={7}>
                              <Text size="xs" c="dimmed" ta="center">
                                No overlapping classes between this model and the test split.
                              </Text>
                            </Table.Td>
                          </Table.Tr>
                        )}
                      </Table.Tbody>
                    </Table>
                  </Table.ScrollContainer>
                </Card>
              ))}
            </SimpleGrid>
          </Stack>

          {/* visual comparison */}
          <Stack gap="xs">
            <Group gap="md">
              <Text size="sm" fw={600}>
                Per-image boxes
              </Text>
              <Group gap={6}>
                <span style={{ width: 14, height: 0, borderTop: `2px dashed ${GT_COLOR}` }} />
                <Text size="xs" c="dimmed">
                  Ground truth
                </Text>
                {data.per_entry.map((e) => (
                  <Group gap={4} key={e.entry_id}>
                    <span style={{ width: 14, height: 2, background: colorOf(e.entry_id) }} />
                    <Text size="xs" c="dimmed">
                      {labelOf(e)}
                    </Text>
                  </Group>
                ))}
              </Group>
            </Group>
            <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
              {data.images.map((img) => (
                <Stack
                  key={img.stem}
                  gap={4}
                  style={{ cursor: 'zoom-in' }}
                  onClick={() => setEnlarged(img)}
                >
                  <BoxOverlay
                    src={img.url}
                    layers={[
                      { boxes: img.gt_boxes, color: GT_COLOR, dashed: true },
                      ...img.per_entry.map((pe) => ({
                        boxes: pe.pred_boxes,
                        color: colorOf(pe.entry_id),
                        dashed: false,
                      })),
                    ]}
                  />
                  <Text size="xs" c="dimmed" truncate="end">
                    {img.name}
                  </Text>
                </Stack>
              ))}
            </SimpleGrid>
          </Stack>
        </Stack>
      )}

      <Modal opened={!!enlarged} onClose={() => setEnlarged(null)} size="xl" title={enlarged?.name}>
        {enlarged && (
          <BoxOverlay
            src={enlarged.url}
            layers={[
              { boxes: enlarged.gt_boxes, color: GT_COLOR, dashed: true },
              ...enlarged.per_entry.map((pe) => ({
                boxes: pe.pred_boxes,
                color: colorOf(pe.entry_id),
                dashed: false,
              })),
            ]}
          />
        )}
      </Modal>
    </Stack>
  )
}

function Metric({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <Stack gap={0} align="center">
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text size={strong ? 'md' : 'sm'} fw={strong ? 700 : 600}>
        {value}
      </Text>
    </Stack>
  )
}

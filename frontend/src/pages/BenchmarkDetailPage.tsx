// 벤치마크 결과 상세 — 텐서보드 결의 화면. 왼쪽 레일에서 엔트리를 켜고 끄고
// conf 슬라이더로 동작점을 옮기면, 오른쪽 탭의 카드들이 그 선택만 그린다.
//
// 키는 모델이 아니라 **엔트리**다: 같은 모델이 방식만 달리해 두 번 들어올 수 있어
// `model_id` 로는 구분되지 않는다.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
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
  Slider,
  Stack,
  Table,
  Tabs,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { BarChart, LineChart } from '@mantine/charts'
import { IconAlertTriangle, IconArrowLeft } from '@tabler/icons-react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import ChartCard from '../components/charts/ChartCard'
import ChartGrid from '../components/charts/ChartGrid'
import ConfusionMatrix from '../components/charts/ConfusionMatrix'
import SeriesRail from '../components/charts/SeriesRail'
import {
  ApiError,
  getBenchmarkResult,
  listBenchmarks,
  subscribeBenchmarkEvents,
  type ClassMetric,
  type CompareBox,
  type CompareEntryResult,
  type CompareImage,
  type CompareProgress,
  type OperatingPoint,
} from '../api/client'

const GT_COLOR = '#51cf66' // ground truth = green (dashed)
const ENTRY_COLORS = ['#4dabf7', '#f783ac', '#ffa94d', '#845ef7', '#38d9a9', '#ff8787']

/** 곡선은 언제나 IoU 0.5 에서 계산한다 — PR·F1 곡선의 관례다. 동작점 스냅샷은
 *  런에 설정된 매칭 IoU 를 쓰므로 두 패널의 숫자가 갈릴 수 있고, 그건 버그가 아니다.
 *  카드 제목에 이 값을 박아 두어 차이가 의도된 것으로 읽히게 한다. */
const CURVE_IOU_LABEL = 'IoU 0.50'

/** 정답이 하나도 없는 구간은 결과에서 아예 빠진다 — 세 칸이 다 있다고 볼 수 없다. */
const SIZE_KEYS = ['small', 'medium', 'large'] as const

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

const fmt = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(3))

/** 파라미터 수를 사람이 읽는 자릿수로. 없으면 `—`. */
const fmtParams = (v: number | null | undefined) => {
  if (v == null) return '—'
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`
  return String(v)
}

/** 엔트리를 규정하는 것은 모델 이름 + 방식 + 그 방식의 크기다 —
 *  같은 모델을 `tiled 640` 과 `tiled 512` 로 두 번 넣어도 카드가 구분돼야 한다. */
const entryTitle = (e: CompareEntryResult) => {
  // 크기를 싣기 전에 돌린 옛 런은 이 값이 없다 — 그때는 방식만 보인다.
  const size = e.mode === 'tiled' ? e.tile_size : e.imgsz
  // 타일은 자른 크기와 **타일을 넣는 추론 크기**가 갈릴 수 있고, 그 차이가 점수를
  // 바꾼다. 다를 때만 `@imgsz` 를 덧붙여 크기가 같은 보통의 제목은 조용히 둔다.
  const at = e.mode === 'tiled' && e.imgsz && e.imgsz !== e.tile_size ? ` @${e.imgsz}` : ''
  return `${e.name} · ${e.mode}${size ? ` ${size}` : ''}${at}`
}

/** 곡선들을 하나의 x 격자에 얹는다 — LineChart 가 계열마다 다른 x 를 못 받는다.
 *  각 계열은 자기 점들 중 그 x 이하의 마지막 값을 쓴다(계단 보간). */
function mergeCurves(series: { key: string; points: [number, number][] }[], gridSize = 101) {
  const grid = Array.from({ length: gridSize }, (_, i) => i / (gridSize - 1))
  return grid.map((x) => {
    const row: Record<string, number> = { x: Number(x.toFixed(3)) }
    for (const s of series) {
      let v: number | undefined
      for (const [px, py] of s.points) {
        if (px <= x) v = py
        else break
      }
      if (v !== undefined) row[s.key] = v
    }
    return row
  })
}

/** 데이터셋 test 분할로 채점한 벤치마크 한 건의 결과. 결과가 아직 없으면(404)
 *  진행률을 구독해 기다리고, 오래전에 끝난 런은 구독 없이 바로 결과를 읽는다. */
export default function BenchmarkDetailPage() {
  const { projectId = '', benchId = '' } = useParams()
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [progress, setProgress] = useState<CompareProgress | null>(null)
  const [enlarged, setEnlarged] = useState<CompareImage | null>(null)
  const [enabledIds, setEnabledIds] = useState<Set<string>>(new Set())
  const [confIdx, setConfIdx] = useState(0)
  const [tab, setTab] = useState<string | null>('scalars')
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

  // 파라미터만 바뀌는 이동에서는 React Router 가 컴포넌트를 재사용한다 —
  // 앞 런의 진행률·확대 이미지·일회성 재조회 플래그가 다음 런으로 새어 나가지 않게 지운다.
  // layout effect 인 이유 — 그린 뒤에 지우면 앞 런의 진행 막대·확대 모달이 한 프레임 비친다.
  // 아래 안전망(passive effect)보다 먼저 도는 것도 그대로다: layout effect 는 commit 단계에서
  // passive effect 전체보다 앞서 돌아, 플래그는 언제나 읽히기 전에 지워진다.
  useLayoutEffect(() => {
    refetchedOnDone.current = false
    setProgress(null)
    setEnlarged(null)
    setEnabledIds(new Set())
    setConfIdx(0)
  }, [benchId])

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
  // 색의 출처는 여기 하나뿐이다 — 레일과 차트가 같은 표를 본다.
  const view = useMemo(() => {
    const colors: Record<string, string> = {}
    const labels: Record<string, string> = {}
    const seen = new Map<string, number>()
    ;(data?.per_entry ?? []).forEach((e, i) => {
      colors[e.entry_id] = ENTRY_COLORS[i % ENTRY_COLORS.length]
      const base = entryTitle(e)
      const n = (seen.get(base) ?? 0) + 1
      seen.set(base, n)
      labels[e.entry_id] = n > 1 ? `${base} #${n}` : base
    })
    return { colors, labels }
  }, [data])

  const colorOf = (entryId: string) => view.colors[entryId] ?? '#868e96'
  const labelOf = (e: CompareEntryResult) => view.labels[e.entry_id] ?? e.name

  const entries = useMemo(() => data?.per_entry ?? [], [data])

  // 결과가 오면 전부 켠 상태로 시작한다. 이벤트가 아니라 **데이터**에 매달려 있어
  // 오래전에 끝난 런도(이벤트가 한 번도 오지 않아도) 그대로 켜진다.
  useLayoutEffect(() => {
    setEnabledIds(new Set(entries.map((e) => e.entry_id)))
  }, [entries])

  // 슬라이더 눈금은 첫 엔트리의 동작점 목록을 쓴다 — 워커가 모든 엔트리에 같은
  // conf 격자를 쓰므로 개수와 값이 같다. 값을 **읽을** 때는 엔트리 자기 배열을 쓴다.
  const steps = entries[0]?.operating_points ?? []
  const op = steps.length ? steps[Math.min(confIdx, steps.length - 1)] : undefined

  // 처음 보이는 동작점은 런의 공식 conf 여야 한다 — 화면을 열자마자 목록의
  // 대표 숫자와 다른 값이 보이면 그게 버그로 읽힌다.
  useLayoutEffect(() => {
    const pts = data?.per_entry?.[0]?.operating_points ?? []
    if (!pts.length) return
    const target = data?.conf ?? 0.4
    let best = 0
    pts.forEach((s, i) => {
      if (Math.abs(s.conf - target) < Math.abs(pts[best].conf - target)) best = i
    })
    setConfIdx(best)
  }, [data])

  const shown = entries.filter((e) => enabledIds.has(e.entry_id))

  /** 엔트리 자기 배열에서 슬라이더가 가리키는 동작점. 옛 런은 배열 자체가 없다. */
  const opOf = (e: CompareEntryResult): OperatingPoint | undefined => {
    const pts = e.operating_points
    if (!pts?.length) return undefined
    return pts[Math.min(confIdx, pts.length - 1)]
  }

  // 옛 런에는 없는 지표들이다 — 데이터가 없는 탭은 아예 띄우지 않는다.
  const hasCurves = entries.some((e) => (e.curves?.pr?.length ?? 0) > 0)
  const hasConfusion = entries.some((e) => (e.operating_points?.length ?? 0) > 0)
  const hasSize = entries.some((e) => e.by_size && Object.keys(e.by_size).length > 0)
  const hasSpeed = entries.some((e) => e.speed)

  // 앞 런에서 고른 탭이 이 런에는 없을 수 있다 — 그때는 빈 화면 대신 Scalars 로.
  const available = new Set(['scalars', 'images'])
  if (hasCurves) available.add('curves')
  if (hasConfusion) available.add('matrix')
  const activeTab = tab && available.has(tab) ? tab : 'scalars'

  // "Best" = highest mAP@0.5:0.95 (the headline COCO metric)
  const bestId = useMemo(() => {
    if (!data?.per_entry.length) return null
    return data.per_entry.reduce((a, b) => (b.map > a.map ? b : a)).entry_id
  }, [data])

  /** 켜진 엔트리 × 클래스로 곡선 계열을 만든다. 색은 엔트리 색을 그대로 쓰고,
   *  클래스가 여럿인 엔트리만 이름을 덧붙인다. */
  const curveSeries = (which: 'pr' | 'f1_conf') => {
    const out: { key: string; color: string; points: [number, number][] }[] = []
    for (const e of shown) {
      const cs = e.curves?.[which] ?? []
      for (const c of cs) {
        out.push({
          key: cs.length > 1 ? `${labelOf(e)} · ${c.name}` : labelOf(e),
          color: colorOf(e.entry_id),
          points: c.points,
        })
      }
    }
    return out
  }

  const barSeriesOf = (list: CompareEntryResult[]) =>
    list.map((e) => ({ name: labelOf(e), color: colorOf(e.entry_id) }))

  // AP by size — 세 칸을 따로따로 확인한다. 정답이 없는 구간은 키가 아예 없다.
  const sizeEntries = shown.filter((e) => e.by_size && Object.keys(e.by_size).length > 0)
  const sizeData = SIZE_KEYS.filter((k) => sizeEntries.some((e) => e.by_size?.[k])).map((k) => {
    const row: Record<string, string | number> = { size: k }
    for (const e of sizeEntries) {
      const b = e.by_size?.[k]
      if (b) row[labelOf(e)] = b.ap50
    }
    return row
  })

  const speedEntries = shown.filter((e) => e.speed)
  const speedData = speedEntries.length
    ? [
        {
          metric: 'ms/img',
          ...Object.fromEntries(speedEntries.map((e) => [labelOf(e), e.speed?.ms_median ?? 0])),
        },
      ]
    : []

  const notReady = result.error instanceof ApiError && result.error.status === 404
  const resultError = result.error && !notReady ? (result.error as Error).message : null
  const conf = bench?.conf ?? data?.conf
  const iou = bench?.iou ?? data?.iou
  const entryCount = data?.per_entry.length ?? bench?.entries
  // 진행 막대는 **진짜 도는 동안만**. 오래전에 끝난 런을 여는 것이 이 화면의 기본
  // 경로인데, 결과 JSON 을 받는 동안 0% 막대를 띄우면 방금 시작한 것처럼 보인다.
  const waiting = !data && running
  const loading = !data && !running && (benchmarks.isLoading || result.isFetching)

  return (
    <Stack gap="md">
      {/* header */}
      <Group justify="space-between" wrap="wrap">
        <Group gap="xs">
          <Tooltip label="Back to Benchmarks">
            <ActionIcon variant="default" onClick={() => navigate(`/projects/${projectId}/models?tab=compare`)}>
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

      {/* 목록을 **받아 온 뒤에도** 없을 때만 "지워졌다"고 말한다 — 조회 실패는
          `data: undefined` 라 이 조건에 섞이면 살아 있는 런을 삭제된 것으로 만든다 */}
      {benchmarks.isSuccess && !bench && (
        <Alert color="gray" icon={<IconAlertTriangle size={18} />}>
          This benchmark is no longer in the history.
        </Alert>
      )}
      {benchmarks.isError && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />}>
          Could not load the benchmark list: {(benchmarks.error as Error).message}
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
      {!data && !waiting && !loading && notReady && bench?.status !== 'running' && (
        <Alert color="gray" icon={<IconAlertTriangle size={18} />}>
          No result was stored for this benchmark.
        </Alert>
      )}
      {resultError && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />}>
          {resultError}
        </Alert>
      )}

      {loading && (
        <Group gap="sm" justify="center" py="xl">
          <Loader size="md" />
          <Text size="sm" c="dimmed">
            Loading results…
          </Text>
        </Group>
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

          <Group align="flex-start" gap="md" wrap="nowrap">
            <SeriesRail
              title="Entries"
              series={entries.map((e) => ({
                id: e.entry_id,
                label: labelOf(e),
                color: colorOf(e.entry_id),
                hint: e.speed
                  ? `${e.speed.ms_median.toFixed(1)} ms${e.speed.fps ? ` · ${e.speed.fps.toFixed(1)} fps` : ''}`
                  : undefined,
              }))}
              enabled={enabledIds}
              onToggle={(id) =>
                setEnabledIds((prev) => {
                  const next = new Set(prev)
                  if (next.has(id)) next.delete(id)
                  else next.add(id)
                  return next
                })
              }
            >
              {steps.length > 0 && (
                <>
                  <Text size="xs" fw={700} c="dimmed" tt="uppercase" mb={4}>
                    Confidence
                  </Text>
                  <Slider
                    min={0}
                    max={steps.length - 1}
                    step={1}
                    value={Math.min(confIdx, steps.length - 1)}
                    onChange={setConfIdx}
                    label={(i) => steps[i]?.conf.toFixed(2) ?? ''}
                  />
                  <Text size="xs" c="dimmed" mt={4}>
                    {op?.conf.toFixed(2)}
                    {op != null &&
                      data.conf != null &&
                      Math.abs(op.conf - data.conf) < 1e-9 &&
                      ' · benchmark default'}
                  </Text>
                </>
              )}
            </SeriesRail>

            <div style={{ flex: 1, minWidth: 0 }}>
              <Tabs value={activeTab} onChange={setTab} keepMounted={false}>
                <Tabs.List mb="md">
                  <Tabs.Tab value="scalars">Scalars</Tabs.Tab>
                  {hasCurves && <Tabs.Tab value="curves">Curves</Tabs.Tab>}
                  {hasConfusion && <Tabs.Tab value="matrix">Matrix</Tabs.Tab>}
                  <Tabs.Tab value="images">Images</Tabs.Tab>
                </Tabs.List>

                {/* ---------- Scalars ---------- */}
                <Tabs.Panel value="scalars">
                  <Stack gap="md">
                    <Card withBorder radius="md" padding="sm">
                      <Table.ScrollContainer minWidth={760}>
                        <Table striped highlightOnHover fz="xs" verticalSpacing={6}>
                          <Table.Thead>
                            <Table.Tr>
                              <Table.Th>Entry</Table.Th>
                              <Table.Th ta="right">mAP50</Table.Th>
                              <Table.Th ta="right">mAP50-95</Table.Th>
                              <Table.Th ta="right">AP75</Table.Th>
                              <Table.Th ta="right">P</Table.Th>
                              <Table.Th ta="right">R</Table.Th>
                              <Table.Th ta="right">F1</Table.Th>
                              <Table.Th ta="right">Detections</Table.Th>
                              <Table.Th ta="right">ms/img</Table.Th>
                              <Table.Th ta="right">Params</Table.Th>
                            </Table.Tr>
                          </Table.Thead>
                          <Table.Tbody>
                            {shown.map((e) => {
                              // 동작점이 있으면 슬라이더가 가리키는 값을, 옛 런은
                              // 결과에 박힌 공식 동작점의 값을 그대로 쓴다.
                              const o = opOf(e)?.overall ?? e.overall
                              return (
                                <Table.Tr key={e.entry_id}>
                                  <Table.Td>
                                    <Group gap={6} wrap="nowrap">
                                      <span
                                        style={{
                                          width: 10,
                                          height: 10,
                                          borderRadius: 2,
                                          background: colorOf(e.entry_id),
                                          flexShrink: 0,
                                        }}
                                      />
                                      <Text size="xs" fw={600}>
                                        {labelOf(e)}
                                      </Text>
                                      {e.entry_id === bestId && (
                                        <Badge size="xs" color="teal" variant="light">
                                          Best mAP
                                        </Badge>
                                      )}
                                    </Group>
                                  </Table.Td>
                                  <Table.Td ta="right">{fmt(e.map50)}</Table.Td>
                                  <Table.Td ta="right">{fmt(e.map)}</Table.Td>
                                  <Table.Td ta="right">{fmt(e.ap75)}</Table.Td>
                                  <Table.Td ta="right">{fmt(o.precision)}</Table.Td>
                                  <Table.Td ta="right">{fmt(o.recall)}</Table.Td>
                                  <Table.Td ta="right">{fmt(o.f1)}</Table.Td>
                                  <Table.Td ta="right">{e.detections}</Table.Td>
                                  <Table.Td ta="right">
                                    {e.speed ? e.speed.ms_median.toFixed(1) : '—'}
                                  </Table.Td>
                                  <Table.Td ta="right">{fmtParams(e.model?.params)}</Table.Td>
                                </Table.Tr>
                              )
                            })}
                            {shown.length === 0 && (
                              <Table.Tr>
                                <Table.Td colSpan={10}>
                                  <Text size="xs" c="dimmed" ta="center">
                                    No entries selected. Turn one on in the rail.
                                  </Text>
                                </Table.Td>
                              </Table.Tr>
                            )}
                          </Table.Tbody>
                        </Table>
                      </Table.ScrollContainer>
                    </Card>

                    {(hasSize || hasSpeed) && (
                      <ChartGrid>
                        {hasSize && sizeData.length > 0 && (
                          <ChartCard title="AP by object size" hint="AP@0.50">
                            <BarChart
                              h={240}
                              data={sizeData}
                              dataKey="size"
                              series={barSeriesOf(sizeEntries)}
                              yAxisProps={{ domain: [0, 1] }}
                              withLegend
                            />
                          </ChartCard>
                        )}
                        {hasSpeed && speedData.length > 0 && (
                          <ChartCard title="Inference speed" hint="ms/img (lower is better)">
                            <BarChart
                              h={240}
                              data={speedData}
                              dataKey="metric"
                              series={barSeriesOf(speedEntries)}
                              withLegend
                            />
                          </ChartCard>
                        )}
                      </ChartGrid>
                    )}

                    {/* per-class metrics, one table per entry */}
                    <Stack gap="xs">
                      <Text size="sm" fw={600}>
                        Per-class metrics
                      </Text>
                      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="sm">
                        {shown.map((e) => {
                          // P/R/F1 은 동작점을 따라 움직이고, AP 는 conf 와 무관한
                          // 곡선 아래 면적이라 엔트리 본체의 값을 그대로 쓴다.
                          const byCls = new Map<number, ClassMetric>()
                          for (const c of opOf(e)?.per_class ?? []) byCls.set(c.cls, c)
                          return (
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
                                    {e.per_class.map((c) => {
                                      const at = byCls.get(c.cls) ?? c
                                      return (
                                        <Table.Tr key={c.cls}>
                                          <Table.Td>{c.name}</Table.Td>
                                          <Table.Td ta="right">{c.gt}</Table.Td>
                                          <Table.Td ta="right">{fmt(at.precision)}</Table.Td>
                                          <Table.Td ta="right">{fmt(at.recall)}</Table.Td>
                                          <Table.Td ta="right">{fmt(at.f1)}</Table.Td>
                                          <Table.Td ta="right">{fmt(c.ap50)}</Table.Td>
                                          <Table.Td ta="right">{fmt(c.ap)}</Table.Td>
                                        </Table.Tr>
                                      )
                                    })}
                                    {e.per_class.length === 0 && (
                                      <Table.Tr>
                                        <Table.Td colSpan={7}>
                                          <Text size="xs" c="dimmed" ta="center">
                                            No overlapping classes between this model and the test
                                            split.
                                          </Text>
                                        </Table.Td>
                                      </Table.Tr>
                                    )}
                                  </Table.Tbody>
                                </Table>
                              </Table.ScrollContainer>
                            </Card>
                          )
                        })}
                      </SimpleGrid>
                    </Stack>
                  </Stack>
                </Tabs.Panel>

                {/* ---------- Curves ---------- */}
                {hasCurves && (
                  <Tabs.Panel value="curves">
                    <CurvesPanel
                      pr={curveSeries('pr')}
                      f1={curveSeries('f1_conf')}
                      bestF1={shown
                        .map((e) => {
                          const b = e.curves?.best_f1
                          if (!b) return null
                          const who = shown.length > 1 ? `${labelOf(e)}: ` : ''
                          return `${who}best F1 ${b.value.toFixed(2)} @ ${b.conf.toFixed(2)}`
                        })
                        .filter((s): s is string => s !== null)
                        .join(' · ')}
                    />
                  </Tabs.Panel>
                )}

                {/* ---------- Matrix ---------- */}
                {hasConfusion && (
                  <Tabs.Panel value="matrix">
                    <ChartGrid>
                      {shown.map((e) => {
                        const point = opOf(e)
                        if (!point?.confusion) return null
                        return (
                          <ChartCard
                            key={e.entry_id}
                            title={labelOf(e)}
                            hint={`@ conf ${point.conf.toFixed(2)}`}
                          >
                            <ConfusionMatrix
                              labels={point.confusion.labels}
                              rows={point.confusion.rows}
                            />
                          </ChartCard>
                        )
                      })}
                    </ChartGrid>
                  </Tabs.Panel>
                )}

                {/* ---------- Images ---------- */}
                <Tabs.Panel value="images">
                  <Stack gap="xs">
                    <Group gap="md">
                      <Group gap={6}>
                        <span style={{ width: 14, height: 0, borderTop: `2px dashed ${GT_COLOR}` }} />
                        <Text size="xs" c="dimmed">
                          Ground truth
                        </Text>
                        {shown.map((e) => (
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
                              ...img.per_entry
                                .filter((pe) => enabledIds.has(pe.entry_id))
                                .map((pe) => ({
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
                </Tabs.Panel>
              </Tabs>
            </div>
          </Group>
        </Stack>
      )}

      <Modal opened={!!enlarged} onClose={() => setEnlarged(null)} size="xl" title={enlarged?.name}>
        {enlarged && (
          <BoxOverlay
            src={enlarged.url}
            layers={[
              { boxes: enlarged.gt_boxes, color: GT_COLOR, dashed: true },
              ...enlarged.per_entry
                .filter((pe) => enabledIds.has(pe.entry_id))
                .map((pe) => ({
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

/** PR 곡선과 F1–conf 곡선. 둘 다 **IoU 0.5** 에서 계산된 값이라 제목에 그 사실을 박는다 —
 *  런의 매칭 IoU 가 0.5 가 아니면 Scalars 탭의 숫자와 어긋나 보이기 때문이다. */
function CurvesPanel({
  pr,
  f1,
  bestF1,
}: {
  pr: { key: string; color: string; points: [number, number][] }[]
  f1: { key: string; color: string; points: [number, number][] }[]
  bestF1: string
}) {
  const prData = mergeCurves(pr)
  const f1Data = mergeCurves(f1)
  const asSeries = (list: { key: string; color: string }[]) =>
    list.map((s) => ({ name: s.key, color: s.color }))

  return (
    <ChartGrid>
      <ChartCard title={`PR curve · ${CURVE_IOU_LABEL}`} hint="x: recall · y: precision">
        {pr.length > 0 ? (
          <LineChart
            h={260}
            data={prData}
            dataKey="x"
            series={asSeries(pr)}
            curveType="monotone"
            withDots={false}
            withLegend
            connectNulls
            xAxisProps={{ type: 'number', domain: [0, 1], tickCount: 6 }}
            yAxisProps={{ domain: [0, 1] }}
            valueFormatter={(v) => v.toFixed(3)}
          />
        ) : (
          <Text size="xs" c="dimmed">
            No entry selected.
          </Text>
        )}
      </ChartCard>

      <ChartCard
        title={`F1 – confidence · ${CURVE_IOU_LABEL}`}
        hint={bestF1 || 'x: confidence · y: F1'}
      >
        {f1.length > 0 ? (
          <LineChart
            h={260}
            data={f1Data}
            dataKey="x"
            series={asSeries(f1)}
            curveType="monotone"
            withDots={false}
            withLegend
            connectNulls
            xAxisProps={{ type: 'number', domain: [0, 1], tickCount: 6 }}
            yAxisProps={{ domain: [0, 1] }}
            valueFormatter={(v) => v.toFixed(3)}
          />
        ) : (
          <Text size="xs" c="dimmed">
            No entry selected.
          </Text>
        )}
      </ChartCard>
    </ChartGrid>
  )
}

// 학습 런 상세 — 텐서보드 결의 화면. 왼쪽 레일에서 지표 계열을 켜고 끄고 스무딩을
// 조절하면, 오른쪽 탭의 카드들이 그 선택만 그린다.
//
// 접이식 토글 넷(loss·클래스 지표·결과 이미지·로그)을 걷어낸 자리다. 하나를 펼쳐
// 스크롤하면 다른 것에 닿으려고 위로 되돌아가야 했다 — 탭이 그 왕복을 없앤다.

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Code,
  Group,
  Image,
  Loader,
  Modal,
  ScrollArea,
  SimpleGrid,
  Slider,
  Stack,
  Tabs,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import {
  ChartLegend,
  LineChart,
} from '@mantine/charts'
import {
  IconArrowLeft,
  IconDownload,
  IconPlaylistAdd,
} from '@tabler/icons-react'
import {
  notifications,
} from '@mantine/notifications'
import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import {
  useNavigate,
  useParams,
} from 'react-router-dom'
import {
  api,
  getRunLog,
  getRunPerClass,
  getRunPerClassHistory,
  getRunResults,
  runArgsYamlUrl,
  runResultsCsvUrl,
  subscribeTrainEvents,
  type ModelOut,
  type TrainEpochEvent,
  type TrainRunOut,
} from '../api/client'
import StatTile from '../components/StatTile'
import ChartCard from '../components/charts/ChartCard'
import ChartGrid from '../components/charts/ChartGrid'
import SeriesRail from '../components/charts/SeriesRail'
import { smoothSeries } from '../components/charts/smoothing'
import DetailsCard from '../components/train/DetailsCard'
import PerClassEpochChart from '../components/train/PerClassEpochChart'
import PerClassTable from '../components/train/PerClassTable'
import PlotsSection from '../components/train/PlotsSection'
import {
  RUN_STATUS_COLOR,
  eventToPoint,
  formatDuration,
  rowToPoint,
  type Point,
} from '../components/train/metrics'


type SeriesKey =
  | 'mAP50'
  | 'mAP50-95'
  | 'precision'
  | 'recall'
  | 'train_box'
  | 'train_cls'
  | 'train_dfl'
  | 'val_box'
  | 'val_cls'
  | 'val_dfl'
  | 'lr'

interface SeriesDef {
  key: SeriesKey
  /** 차트 범례에 올리는 이름. 카드 하나가 한 묶음이라 묶음 안에서만 구분되면 된다. */
  label: string
  color: string
}

/** 계열 정의의 출처는 여기 하나뿐이다 — 레일 견본과 선이 같은 색을 쓰게 하려면
 *  색이 한 곳에만 적혀 있어야 한다. 값은 접이식 차트가 쓰던 것을 그대로 옮겼고,
 *  `var(--mantine-color-*)` 로 적어 레일의 `background` 에도 그대로 꽂힌다.
 *  묶음마다 색이 되풀이되지만 카드 하나가 한 묶음이라 한 축에서 겹치지 않는다. */
const METRIC_SERIES: SeriesDef[] = [
  { key: 'mAP50', label: 'mAP50', color: 'var(--mantine-color-teal-6)' },
  { key: 'mAP50-95', label: 'mAP50-95', color: 'var(--mantine-color-blue-6)' },
]
const PR_SERIES: SeriesDef[] = [
  { key: 'precision', label: 'precision', color: 'var(--mantine-color-indigo-6)' },
  { key: 'recall', label: 'recall', color: 'var(--mantine-color-orange-6)' },
]
const TRAIN_LOSS_SERIES: SeriesDef[] = [
  { key: 'train_box', label: 'box', color: 'var(--mantine-color-teal-6)' },
  { key: 'train_cls', label: 'cls', color: 'var(--mantine-color-orange-6)' },
  { key: 'train_dfl', label: 'dfl', color: 'var(--mantine-color-grape-6)' },
]
const VAL_LOSS_SERIES: SeriesDef[] = [
  { key: 'val_box', label: 'box', color: 'var(--mantine-color-teal-6)' },
  { key: 'val_cls', label: 'cls', color: 'var(--mantine-color-orange-6)' },
  { key: 'val_dfl', label: 'dfl', color: 'var(--mantine-color-grape-6)' },
]
const LR_SERIES: SeriesDef[] = [
  { key: 'lr', label: 'lr', color: 'var(--mantine-color-gray-6)' },
]

const ALL_SERIES: SeriesDef[] = [
  ...METRIC_SERIES,
  ...PR_SERIES,
  ...TRAIN_LOSS_SERIES,
  ...VAL_LOSS_SERIES,
  ...LR_SERIES,
]

/** 레일 이름표는 **데이터 키 그대로** 쓴다 — 차트 범례의 `box` 는 카드 제목이
 *  묶음을 말해 주지만, 레일에는 세 묶음이 한 줄로 섞여 `train_box` 와 `val_box` 가
 *  구분돼야 하기 때문이다. */
const railLabel = (key: SeriesKey) => key

/** 처음 보이는 것은 지금까지 늘 보이던 넷 — 화면의 첫인상을 바꾸지 않는다. */
const DEFAULT_ENABLED: SeriesKey[] = ['mAP50', 'mAP50-95', 'precision', 'recall']

/** 스무딩을 켜면 계열마다 선이 둘이다. 원본 값은 이 꼬리표를 단 별도 열로 옮기고
 *  원래 키에는 스무딩한 값을 둔다 — 범례·툴팁이 가리키는 쪽이 진한 선이 되게. */
const RAW_SUFFIX = '__raw'

const isRaw = (name: string) => name.endsWith(RAW_SUFFIX)

/** 원본 선은 흐리게, 점 없이. `fill: 'none'` 은 그리기와 무관하고(recharts 의 선은
 *  언제나 fill 이 none 이다) Mantine 툴팁이 이 항목을 걸러 내게 하는 표식이다 —
 *  한 에폭에 같은 이름이 두 줄로 뜨면 읽을 수 없다. */
const rawLineProps = (s: { name: string }) =>
  isRaw(s.name)
    ? { strokeOpacity: 0.25, strokeWidth: 1.5, dot: false, activeDot: false, fill: 'none' }
    : {}

/** 범례에는 스무딩된 선만 올린다. Mantine 이 `legendProps` 를 recharts `Legend` 에
 *  그대로 넘겨 주므로 `content` 만 갈아 끼워 원본 항목을 걸러 낸다. */
function SmoothedLegend({
  payload,
  series,
}: {
  payload?: readonly Record<string, unknown>[]
  series: { name: string; label: string; color: string }[]
}) {
  return (
    <ChartLegend
      payload={payload?.filter((p) => !isRaw(String(p.dataKey)))}
      series={series}
      onHighlight={() => {}}
      legendPosition="top"
    />
  )
}

/** 켜진 계열이 하나도 없으면 카드째 사라진다 — 빈 축을 그리지 않는다. */
function ScalarCard({
  title,
  hint,
  defs,
  data,
  smoothing,
  withDots,
  withLegend = true,
  yDomain,
  referenceLines,
  valueFormatter,
}: {
  title: string
  hint?: string
  defs: SeriesDef[]
  data: Record<string, number | undefined>[]
  smoothing: number
  withDots: boolean
  withLegend?: boolean
  yDomain?: [number, number]
  referenceLines?: { x: number; label: string; color: string }[]
  valueFormatter: (v: number) => string
}) {
  if (!defs.length) return null

  // 원본을 먼저 깔고 스무딩한 선을 그 위에 얹는다 — 순서가 뒤집히면 흐린 선이
  // 진한 선을 덮어 색이 바랜다.
  const series = defs.flatMap((s) =>
    smoothing > 0
      ? [
          { name: `${s.key}${RAW_SUFFIX}`, color: s.color, label: s.label },
          { name: s.key as string, color: s.color, label: s.label },
        ]
      : [{ name: s.key as string, color: s.color, label: s.label }],
  )

  return (
    <ChartCard title={title} hint={hint}>
      <LineChart
        h={220}
        data={data}
        dataKey="epoch"
        series={series}
        curveType="monotone"
        withDots={withDots}
        withLegend={withLegend}
        legendProps={
          smoothing > 0 ? { content: <SmoothedLegend series={series} /> } : undefined
        }
        lineProps={rawLineProps}
        yAxisProps={yDomain ? { domain: yDomain } : undefined}
        referenceLines={referenceLines}
        valueFormatter={valueFormatter}
      />
    </ChartCard>
  )
}

export default function TrainRunDetailPage() {
  const { projectId = '', runId = '' } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // 워커가 `done` 을 찍는 것과 API 가 행을 done 으로 바꾸는 것 사이에 틈이 있다 —
  // SSE 로 무효화한 재조회가 그 틈에 걸리면 `running` 인 응답을 받고, 스트림은 이미
  // 닫혀 다시 깨울 것이 없다. 도는 동안은 폴링해 두어 반드시 따라잡게 한다.
  const run = useQuery({
    queryKey: ['train-run', runId],
    queryFn: () => api.get<TrainRunOut>(`/training/runs/${runId}`),
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 4000 : false),
  })
  const r = run.data
  const running = r?.status === 'running'

  const results = useQuery({
    queryKey: ['train-results', runId],
    queryFn: () => getRunResults(runId),
    refetchInterval: running ? 4000 : false,
  })

  const artifacts = useQuery({
    queryKey: ['train-artifacts', runId],
    queryFn: () =>
      api.get<{ files: { name: string; url: string }[]; weights: { name: string; url: string }[] }>(
        `/training/runs/${runId}/artifacts`,
      ),
  })

  const log = useQuery({
    queryKey: ['train-log', runId],
    queryFn: () => getRunLog(runId),
    refetchInterval: running ? 3000 : false,
  })

  const perClass = useQuery({
    queryKey: ['train-per-class', runId],
    queryFn: () => getRunPerClass(runId),
    refetchInterval: running ? 8000 : false,
  })

  const perClassHistory = useQuery({
    queryKey: ['train-per-class-history', runId],
    queryFn: () => getRunPerClassHistory(runId),
    refetchInterval: running ? 8000 : false,
  })

  // live status (fast) + fallback points before results.csv exists
  const [liveStatus, setLiveStatus] = useState<TrainEpochEvent | null>(null)
  const [ssePoints, setSsePoints] = useState<Point[]>([])
  const unsubscribe = useRef<(() => void) | null>(null)

  useEffect(() => {
    setLiveStatus(null)
    setSsePoints([])
    unsubscribe.current?.()
    unsubscribe.current = subscribeTrainEvents(runId, (ev) => {
      setLiveStatus(ev)
      const p = eventToPoint(ev)
      if (p) setSsePoints((prev) => (prev.some((x) => x.epoch === p.epoch) ? prev : [...prev, p]))
      // 'tiling' 은 20 장마다 한 번씩, 많으면 수백 번 온다 — 매번 네 개를 무효화하면
      // 그 프레임을 자르는 몇 분 동안 API 를 그만큼 다시 두들긴다. epoch 처럼 뺀다.
      if (ev.phase !== 'epoch' && ev.phase !== 'tiling') {
        queryClient.invalidateQueries({ queryKey: ['train-run', runId] })
        queryClient.invalidateQueries({ queryKey: ['train-runs', projectId] })
        queryClient.invalidateQueries({ queryKey: ['train-results', runId] })
        queryClient.invalidateQueries({ queryKey: ['train-artifacts', runId] })
      }
    })
    return () => unsubscribe.current?.()
  }, [runId]) // eslint-disable-line react-hooks/exhaustive-deps

  const [lightbox, setLightbox] = useState<{ url: string; label: string } | null>(null)
  const [tab, setTab] = useState<string | null>('scalars')
  const [enabled, setEnabled] = useState<Set<string>>(new Set<string>(DEFAULT_ENABLED))
  const [smoothing, setSmoothing] = useState(0.6)

  // a failed run's cause lives in the log — surface it automatically
  useEffect(() => {
    if (r?.status === 'error') setTab('log')
  }, [r?.status])

  const stop = useMutation({
    mutationFn: () => api.post<TrainRunOut>(`/training/runs/${runId}/stop`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['train-run', runId] })
      queryClient.invalidateQueries({ queryKey: ['train-runs', projectId] })
    },
  })

  const register = useMutation({
    mutationFn: (which: string) => api.post<ModelOut>(`/training/runs/${runId}/register`, { which }),
    onSuccess: (m) => {
      notifications.show({ message: `Added to models: ${m.name}`, color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['models', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  // chart data: prefer canonical results.csv; fall back to live SSE points
  const points = useMemo<Point[]>(() => {
    const rows = results.data ?? []
    if (rows.length) return rows.map(rowToPoint).filter((p) => p.epoch > 0)
    return ssePoints
  }, [results.data, ssePoints])

  // 값이 한 번도 온 적 없는 계열은 레일에 올리지 않는다 — 켤 수는 있는데
  // 켜도 아무것도 안 그려지는 줄이 남으면 고장으로 읽힌다.
  const availableKeys = useMemo(
    () => new Set(ALL_SERIES.filter((s) => points.some((p) => p[s.key] != null)).map((s) => s.key)),
    [points],
  )

  // 스무딩이 켜지면 원본을 `__raw` 열로 복사해 두고 원래 키를 스무딩 값으로 덮는다.
  const chartData = useMemo<Record<string, number | undefined>[]>(() => {
    const rows = points.map((p) => ({ ...p }) as Record<string, number | undefined>)
    if (smoothing <= 0) return rows
    for (const s of ALL_SERIES) {
      const raw = points.map((p) => p[s.key])
      const smoothed = smoothSeries(raw, smoothing)
      rows.forEach((row, i) => {
        row[`${s.key}${RAW_SUFFIX}`] = raw[i]
        row[s.key] = smoothed[i]
      })
    }
    return rows
  }, [points, smoothing])

  const best = useMemo(() => {
    let b: Point | null = null
    for (const p of points) {
      const key = p['mAP50-95'] ?? p.mAP50
      const bk = b ? (b['mAP50-95'] ?? b.mAP50) : undefined
      if (key != null && (bk == null || key > bk)) b = p
    }
    return b
  }, [points])
  const last = points.length ? points[points.length - 1] : null
  const f3 = (v?: number) => (v != null ? v.toFixed(3) : '–')

  const livePhase = liveStatus?.phase
  const liveHasEpoch = livePhase === 'epoch' || livePhase === 'epoch_start'
  const epochTotal =
    (liveHasEpoch || livePhase === 'start' ? liveStatus?.epochs : undefined) ??
    Number(r?.params.epochs ?? 0)
  const rawEpoch = liveHasEpoch ? (liveStatus?.epoch ?? 0) : points.length
  const epochNow = epochTotal ? Math.min(rawEpoch, epochTotal) : rawEpoch
  // running but no epoch has begun yet = still loading model / scanning dataset
  const preparing = running && rawEpoch === 0
  // last epoch finished but the run is still alive = final validation + saving weights
  const finalizing = running && livePhase === 'epoch' && epochTotal > 0 && epochNow >= epochTotal
  const stageLabel =
    livePhase === 'start'
      ? 'Starting…'
      : livePhase === 'tiling'
        ? 'Cutting tiles…'
        : livePhase === 'staging'
          ? 'Staging…'
          : livePhase === 'preparing'
            ? 'Preparing…'
            : 'Initializing…'

  const durationSec =
    points[points.length - 1]?.time ??
    (r?.finished_at
      ? (new Date(r.finished_at).getTime() - new Date(r.created_at).getTime()) / 1000
      : undefined)

  const bestEpoch = best?.epoch

  const referenceLines = bestEpoch
    ? [{ x: bestEpoch, label: 'best', color: 'gray.5' }]
    : undefined

  const withDots = points.length < 40
  /** 그 묶음에서 **켜져 있고 값도 있는** 계열만. 하나도 없으면 카드가 사라진다. */
  const shownDefs = (defs: SeriesDef[]) =>
    defs.filter((s) => enabled.has(s.key) && availableKeys.has(s.key))

  const railSeries = ALL_SERIES.filter((s) => availableKeys.has(s.key)).map((s) => ({
    id: s.key as string,
    label: railLabel(s.key),
    color: s.color,
  }))

  // 앞 런에서 고른 탭이 이 런에는 없을 수 있다 — 그때는 빈 화면 대신 Scalars 로.
  const hasPerClass = !!perClassHistory.data?.length || !!perClass.data?.length
  const hasPlots = !!artifacts.data?.files.length
  const availableTabs = new Set(['scalars', 'log'])
  if (hasPerClass) availableTabs.add('per-class')
  if (hasPlots) availableTabs.add('plots')
  const activeTab = tab && availableTabs.has(tab) ? tab : 'scalars'

  return (
    <Stack gap="md">
      {/* header */}
      <Group justify="space-between" wrap="wrap">
        <Group gap="xs">
          <Tooltip label="Back to Training History">
            <ActionIcon variant="default" onClick={() => navigate(`/projects/${projectId}/history`)}>
              <IconArrowLeft size={16} />
            </ActionIcon>
          </Tooltip>
          <Title order={4}>{r?.name ?? runId}</Title>
          {r && (
            <Badge color={RUN_STATUS_COLOR[r.status] ?? 'gray'} variant="light">
              {r.status}
            </Badge>
          )}
          {r?.error && (
            <Text size="sm" c="red">
              {r.error}
            </Text>
          )}
        </Group>
        <Group gap="xs">
          {!!results.data?.length && (
            <Button
              component="a"
              href={runResultsCsvUrl(runId)}
              download
              size="xs"
              variant="default"
              leftSection={<IconDownload size={14} />}
            >
              CSV
            </Button>
          )}
          {r && ['running', 'done', 'stopped'].includes(r.status) && (
            <Button
              component="a"
              href={runArgsYamlUrl(runId)}
              download
              size="xs"
              variant="default"
              leftSection={<IconDownload size={14} />}
            >
              args.yaml
            </Button>
          )}
          {artifacts.data?.weights.map((w) => (
            <Button
              key={w.name}
              component="a"
              href={w.url}
              download
              size="xs"
              variant="default"
              leftSection={<IconDownload size={14} />}
            >
              {w.name}
            </Button>
          ))}
          {(r?.status === 'done' || r?.status === 'stopped') && (
            <Button
              size="xs"
              variant="light"
              leftSection={<IconPlaylistAdd size={14} />}
              onClick={() => register.mutate(r.status === 'done' ? 'best' : 'last')}
              loading={register.isPending}
            >
              Add to Models
            </Button>
          )}
        </Group>
      </Group>

      {/* live progress (only while running) */}
      {r && running && (
        <Card withBorder radius="md" padding="md">
          {preparing ? (
            <Group justify="space-between" wrap="nowrap">
              <Group gap="md" wrap="nowrap">
                <Loader size="sm" />
                <div>
                  <Text size="sm" fw={600}>
                    {stageLabel}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {livePhase === 'tiling' && liveStatus?.total
                      ? `Cutting tiles for train/val — ${liveStatus.done ?? 0} / ${liveStatus.total}`
                      : 'Loading model & scanning the dataset — the first epoch will start shortly.'}
                  </Text>
                </div>
              </Group>
              <Button
                size="xs"
                color="red"
                variant="light"
                onClick={() => stop.mutate()}
                loading={stop.isPending}
              >
                Stop
              </Button>
            </Group>
          ) : (
            <>
              <Group justify="space-between" wrap="nowrap">
                <Group gap="lg">
                  <Loader size="md" />
                  <div>
                    <Text size="sm" fw={600}>
                      {finalizing
                        ? 'Finalizing…'
                        : `Epoch ${epochNow || '–'} / ${epochTotal || '–'}`}
                      {!finalizing && livePhase === 'epoch_start' ? ' · running…' : ''}
                    </Text>
                    {finalizing && (
                      <Text size="xs" c="dimmed">
                        Last epoch done — running final validation & saving weights.
                      </Text>
                    )}
                    <Text size="xs" c="dimmed">
                      {r.base_model_name ?? r.base_model_id} · {r.params.imgsz}px · batch{' '}
                      {r.params.batch} · {r.params.optimizer ?? 'auto'}
                      {r.params.optimizer && r.params.lr0 != null ? ` · lr0 ${r.params.lr0}` : ''}
                    </Text>
                    <Text size="xs" c="dimmed">
                      elapsed {formatDuration(durationSec)}
                    </Text>
                  </div>
                </Group>
                <Button
                  size="xs"
                  color="red"
                  variant="light"
                  onClick={() => stop.mutate()}
                  loading={stop.isPending}
                >
                  Stop
                </Button>
              </Group>
            </>
          )}
        </Card>
      )}

      {/* run details — near the top, below the live progress */}
      {r && <DetailsCard run={r} durationSec={durationSec} />}

      {/* hero: best vs last per metric (big = best.pt, small = last.pt) */}
      {(best || last) && (
        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            Big = <b>best.pt</b>
            {bestEpoch ? ` (epoch ${bestEpoch})` : ''} · small = <b>last.pt</b>
            {last?.epoch ? ` (epoch ${last.epoch})` : ''}
          </Text>
          <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="sm">
            <StatTile
              label="mAP50"
              value={f3(best?.mAP50)}
              sub={last ? `last ${f3(last.mAP50)}` : undefined}
              color="teal"
            />
            <StatTile
              label="mAP50-95"
              value={f3(best?.['mAP50-95'])}
              sub={last ? `last ${f3(last['mAP50-95'])}` : undefined}
              color="blue"
            />
            <StatTile
              label="Precision"
              value={f3(best?.precision)}
              sub={last ? `last ${f3(last.precision)}` : undefined}
              color="indigo"
            />
            <StatTile
              label="Recall"
              value={f3(best?.recall)}
              sub={last ? `last ${f3(last.recall)}` : undefined}
              color="orange"
            />
          </SimpleGrid>
        </Stack>
      )}

      <Group align="flex-start" gap="md" wrap="nowrap">
        {/* 레일은 카드를 **보면서** 만지는 것이라 스크롤해도 옆에 남아야 한다.
            켜고 끄는 것이 Scalars 의 선뿐이므로 다른 탭에서는 걷어 낸다 —
            아무것도 하지 않는 컨트롤을 옆에 세워 두지 않는다. */}
        {activeTab === 'scalars' && railSeries.length > 0 && (
          <div
            style={{
              position: 'sticky',
              top: 16,
              alignSelf: 'flex-start',
              maxHeight: 'calc(100vh - 32px)',
              overflowY: 'auto',
              // 세로 스크롤이 생기면 가로도 auto 가 된다 — 함께 막는다.
              overflowX: 'hidden',
              // 이름이 길어져도 레일이 부풀지 않게 폭을 묶는다.
              flex: '0 0 240px',
              maxWidth: 240,
            }}
          >
            <SeriesRail
              title="Series"
              series={railSeries}
              enabled={enabled}
              onToggle={(id) =>
                setEnabled((prev) => {
                  const next = new Set(prev)
                  if (next.has(id)) next.delete(id)
                  else next.add(id)
                  return next
                })
              }
            >
              <Text size="xs" fw={700} c="dimmed" tt="uppercase" mb={4}>
                Smoothing
              </Text>
              <Slider
                min={0}
                max={0.95}
                step={0.05}
                value={smoothing}
                onChange={setSmoothing}
                label={(v) => v.toFixed(2)}
              />
              <Text size="xs" c="dimmed" mt={4}>
                {smoothing > 0
                  ? `${smoothing.toFixed(2)} — the raw curve stays behind it, faded`
                  : 'off — raw values only'}
              </Text>
            </SeriesRail>
          </div>
        )}

        <div style={{ flex: 1, minWidth: 0 }}>
          <Tabs value={activeTab} onChange={setTab} keepMounted={false}>
            <Tabs.List mb="md">
              <Tabs.Tab value="scalars">Scalars</Tabs.Tab>
              {hasPerClass && <Tabs.Tab value="per-class">Per-class</Tabs.Tab>}
              {hasPlots && <Tabs.Tab value="plots">Plots</Tabs.Tab>}
              <Tabs.Tab value="log">Log</Tabs.Tab>
            </Tabs.List>

            {/* ---------- Scalars ---------- */}
            <Tabs.Panel value="scalars">
              {points.length > 0 ? (
                <ChartGrid>
                  <ScalarCard
                    title="mAP"
                    hint={bestEpoch ? `best @ epoch ${bestEpoch}` : undefined}
                    defs={shownDefs(METRIC_SERIES)}
                    data={chartData}
                    smoothing={smoothing}
                    withDots={withDots}
                    yDomain={[0, 1]}
                    referenceLines={referenceLines}
                    valueFormatter={(v) => v.toFixed(3)}
                  />
                  <ScalarCard
                    title="Precision / Recall"
                    defs={shownDefs(PR_SERIES)}
                    data={chartData}
                    smoothing={smoothing}
                    withDots={withDots}
                    yDomain={[0, 1]}
                    referenceLines={referenceLines}
                    valueFormatter={(v) => v.toFixed(3)}
                  />
                  <ScalarCard
                    title="Train loss"
                    defs={shownDefs(TRAIN_LOSS_SERIES)}
                    data={chartData}
                    smoothing={smoothing}
                    withDots={withDots}
                    valueFormatter={(v) => v.toFixed(3)}
                  />
                  <ScalarCard
                    title="Val loss"
                    defs={shownDefs(VAL_LOSS_SERIES)}
                    data={chartData}
                    smoothing={smoothing}
                    withDots={withDots}
                    referenceLines={referenceLines}
                    valueFormatter={(v) => v.toFixed(3)}
                  />
                  <ScalarCard
                    title="Learning rate"
                    defs={shownDefs(LR_SERIES)}
                    data={chartData}
                    smoothing={smoothing}
                    withDots={false}
                    withLegend={false}
                    valueFormatter={(v) => v.toExponential(1)}
                  />
                </ChartGrid>
              ) : (
                <Text size="sm" c="dimmed">
                  No epoch metrics yet.
                </Text>
              )}
            </Tabs.Panel>

            {/* ---------- Per-class ---------- */}
            {hasPerClass && (
              <Tabs.Panel value="per-class">
                <Stack gap="md">
                  {!!perClassHistory.data?.length && (
                    <PerClassEpochChart history={perClassHistory.data} />
                  )}
                  {/* 클래스가 수십 개면 표만으로 화면을 넘긴다 — 자기 안에서 구르게 한다 */}
                  {!!perClass.data?.length && (
                    <div
                      style={{
                        maxHeight: 'calc(100vh - 420px)',
                        minHeight: 320,
                        overflowY: 'auto',
                        overflowX: 'hidden',
                        paddingRight: 4,
                      }}
                    >
                      <PerClassTable rows={perClass.data} />
                    </div>
                  )}
                </Stack>
              </Tabs.Panel>
            )}

            {/* ---------- Plots ---------- */}
            {hasPlots && (
              <Tabs.Panel value="plots">
                <div
                  style={{
                    maxHeight: 'calc(100vh - 300px)',
                    minHeight: 360,
                    overflowY: 'auto',
                    overflowX: 'hidden',
                    paddingRight: 4,
                  }}
                >
                  <PlotsSection files={artifacts.data?.files ?? []} onOpen={setLightbox} />
                </div>
              </Tabs.Panel>
            )}

            {/* ---------- Log ---------- */}
            {/* 실패한 런의 traceback 이 여기 있다 — 그래서 이 탭은 언제나 있다 */}
            <Tabs.Panel value="log">
              <Card withBorder radius="md" padding="xs">
                {log.data?.truncated && (
                  <Text size="xs" c="dimmed" mb={4}>
                    Showing the last 256&nbsp;KB of the log.
                  </Text>
                )}
                <ScrollArea.Autosize mah="calc(100vh - 320px)" type="auto">
                  <Code block style={{ whiteSpace: 'pre', fontSize: 12, background: 'transparent' }}>
                    {log.data?.text?.trim() || 'No log yet.'}
                  </Code>
                </ScrollArea.Autosize>
              </Card>
            </Tabs.Panel>
          </Tabs>
        </div>
      </Group>

      <Modal
        opened={lightbox !== null}
        onClose={() => setLightbox(null)}
        size="xl"
        title={lightbox?.label}
        centered
      >
        {lightbox && <Image src={lightbox.url} radius="sm" />}
      </Modal>
    </Stack>
  )
}

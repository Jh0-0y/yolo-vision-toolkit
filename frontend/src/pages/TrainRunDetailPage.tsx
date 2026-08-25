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
  Collapse,
  Group,
  Image,
  Loader,
  Modal,
  ScrollArea,
  SimpleGrid,
  Stack,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import {
  LineChart,
} from '@mantine/charts'
import {
  IconArrowLeft,
  IconChevronDown,
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
  const [showDetailCharts, setShowDetailCharts] = useState(false)
  const [showClassMetrics, setShowClassMetrics] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [showLog, setShowLog] = useState(false)

  // a failed run's cause lives in the log — surface it automatically
  useEffect(() => {
    if (r?.status === 'error') setShowLog(true)
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

  const hasLoss = points.some((p) => p.val_box != null || p.train_box != null)
  const hasLr = points.some((p) => p.lr != null)

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

      {/* charts — primary (mAP + P/R) always; loss/LR behind a toggle */}
      {points.length > 0 && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
            <ChartCard title="mAP" hint={bestEpoch ? `best @ epoch ${bestEpoch}` : undefined}>
              <LineChart
                h={220}
                data={points}
                dataKey="epoch"
                series={[
                  { name: 'mAP50', color: 'teal.6' },
                  { name: 'mAP50-95', color: 'blue.6' },
                ]}
                curveType="monotone"
                withDots={points.length < 40}
                withLegend
                yAxisProps={{ domain: [0, 1] }}
                referenceLines={referenceLines}
                valueFormatter={(v) => v.toFixed(3)}
              />
            </ChartCard>

            <ChartCard title="Precision / Recall">
              <LineChart
                h={220}
                data={points}
                dataKey="epoch"
                series={[
                  { name: 'precision', color: 'indigo.6' },
                  { name: 'recall', color: 'orange.6' },
                ]}
                curveType="monotone"
                withDots={points.length < 40}
                withLegend
                yAxisProps={{ domain: [0, 1] }}
                referenceLines={referenceLines}
                valueFormatter={(v) => v.toFixed(3)}
              />
            </ChartCard>
          </SimpleGrid>

          {(hasLoss || hasLr) && (
            <div>
              <Button
                variant="subtle"
                size="compact-sm"
                rightSection={<IconChevronDown size={14} />}
                onClick={() => setShowDetailCharts((v) => !v)}
              >
                {showDetailCharts ? 'Hide' : 'Show'} detailed graphs (loss · learning rate)
              </Button>
              <Collapse expanded={showDetailCharts}>
                <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md" mt="xs">
                  {hasLoss && (
                    <ChartCard title="Train loss">
                      <LineChart
                        h={220}
                        data={points}
                        dataKey="epoch"
                        series={[
                          { name: 'train_box', label: 'box', color: 'teal.6' },
                          { name: 'train_cls', label: 'cls', color: 'orange.6' },
                          { name: 'train_dfl', label: 'dfl', color: 'grape.6' },
                        ]}
                        curveType="monotone"
                        withDots={points.length < 40}
                        withLegend
                        valueFormatter={(v) => v.toFixed(3)}
                      />
                    </ChartCard>
                  )}
                  {hasLoss && (
                    <ChartCard title="Val loss">
                      <LineChart
                        h={220}
                        data={points}
                        dataKey="epoch"
                        series={[
                          { name: 'val_box', label: 'box', color: 'teal.6' },
                          { name: 'val_cls', label: 'cls', color: 'orange.6' },
                          { name: 'val_dfl', label: 'dfl', color: 'grape.6' },
                        ]}
                        curveType="monotone"
                        withDots={points.length < 40}
                        withLegend
                        referenceLines={referenceLines}
                        valueFormatter={(v) => v.toFixed(3)}
                      />
                    </ChartCard>
                  )}
                  {hasLr && (
                    <ChartCard title="Learning rate">
                      <LineChart
                        h={220}
                        data={points}
                        dataKey="epoch"
                        series={[{ name: 'lr', color: 'gray.6' }]}
                        curveType="monotone"
                        withDots={false}
                        valueFormatter={(v) => v.toExponential(1)}
                      />
                    </ChartCard>
                  )}
                </SimpleGrid>
              </Collapse>
            </div>
          )}
        </Stack>
      )}

      {/* class metrics (per-class chart + table) — its own toggle */}
      {(!!perClassHistory.data?.length || !!perClass.data?.length) && (
        <div>
          <Button
            variant="subtle"
            size="compact-sm"
            rightSection={<IconChevronDown size={14} />}
            onClick={() => setShowClassMetrics((v) => !v)}
          >
            {showClassMetrics ? 'Hide' : 'Show'} class metrics (per-class)
          </Button>
          <Collapse expanded={showClassMetrics}>
            <Stack gap="md" mt="xs">
              {!!perClassHistory.data?.length && (
                <PerClassEpochChart history={perClassHistory.data} />
              )}
              {!!perClass.data?.length && <PerClassTable rows={perClass.data} />}
            </Stack>
          </Collapse>
        </div>
      )}

      {/* result plots (curated) + lightbox — behind a toggle */}
      {!!artifacts.data?.files.length && (
        <div>
          <Button
            variant="subtle"
            size="compact-sm"
            rightSection={<IconChevronDown size={14} />}
            onClick={() => setShowResults((v) => !v)}
          >
            {showResults ? 'Hide' : 'Show'} result images (confusion matrix · curves · samples)
          </Button>
          <Collapse expanded={showResults}>
            <div style={{ marginTop: 8 }}>
              <PlotsSection files={artifacts.data.files} onOpen={setLightbox} />
            </div>
          </Collapse>
        </div>
      )}

      {/* raw training log (stdout+stderr) — the place to see failure tracebacks */}
      <div>
        <Button
          variant="subtle"
          size="compact-sm"
          rightSection={<IconChevronDown size={14} />}
          onClick={() => setShowLog((v) => !v)}
        >
          {showLog ? 'Hide' : 'Show'} training log
        </Button>
        <Collapse expanded={showLog}>
          <Card withBorder radius="md" padding="xs" mt={8}>
            {log.data?.truncated && (
              <Text size="xs" c="dimmed" mb={4}>
                Showing the last 256&nbsp;KB of the log.
              </Text>
            )}
            <ScrollArea.Autosize mah={420} type="auto">
              <Code
                block
                style={{ whiteSpace: 'pre', fontSize: 12, background: 'transparent' }}
              >
                {log.data?.text?.trim() || 'No log yet.'}
              </Code>
            </ScrollArea.Autosize>
          </Card>
        </Collapse>
      </div>

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


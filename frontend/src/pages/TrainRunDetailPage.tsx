import { useEffect, useMemo, useRef, useState } from 'react'
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
  MultiSelect,
  Progress,
  ScrollArea,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { LineChart } from '@mantine/charts'
import {
  IconArrowLeft,
  IconChevronDown,
  IconDownload,
  IconPlaylistAdd,
} from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import {
  api,
  getRunLog,
  getRunPerClass,
  getRunPerClassHistory,
  getRunResults,
  subscribeTrainEvents,
  type ModelOut,
  type PerClassEpoch,
  type PerClassRow,
  type TrainEpochEvent,
  type TrainResultRow,
  type TrainRunOut,
} from '../api/client'
import StatTile from '../components/StatTile'
import { classColor } from '../stores/editorStore'

const RUN_STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  stopped: 'yellow',
  queued: 'gray',
}

interface Point {
  epoch: number
  mAP50?: number
  'mAP50-95'?: number
  precision?: number
  recall?: number
  train_box?: number
  train_cls?: number
  train_dfl?: number
  val_box?: number
  val_cls?: number
  val_dfl?: number
  lr?: number
  time?: number
}

const num = (v: unknown): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) ? v : undefined

/** Build a chart point from a results.csv row (keys may vary slightly by version). */
function rowToPoint(row: TrainResultRow): Point {
  const g = (...keys: string[]): number | undefined => {
    for (const k of keys) if (row[k] != null) return num(row[k])
    return undefined
  }
  return {
    epoch: g('epoch') ?? 0,
    mAP50: g('metrics/mAP50(B)', 'metrics/mAP50'),
    'mAP50-95': g('metrics/mAP50-95(B)', 'metrics/mAP50-95'),
    precision: g('metrics/precision(B)', 'metrics/precision'),
    recall: g('metrics/recall(B)', 'metrics/recall'),
    train_box: g('train/box_loss'),
    train_cls: g('train/cls_loss'),
    train_dfl: g('train/dfl_loss'),
    val_box: g('val/box_loss'),
    val_cls: g('val/cls_loss'),
    val_dfl: g('val/dfl_loss'),
    lr: g('lr/pg0', 'lr/pg1', 'lr/pg2'),
    time: g('time'),
  }
}

/** Fallback point from a live SSE event (before results.csv is first flushed). */
function eventToPoint(ev: TrainEpochEvent): Point | null {
  if (ev.phase !== 'epoch' || !ev.epoch || !ev.metrics) return null
  if (ev.epochs && ev.epoch > ev.epochs) return null
  const m = ev.metrics
  return {
    epoch: ev.epoch,
    mAP50: num(m['metrics/mAP50(B)']),
    'mAP50-95': num(m['metrics/mAP50-95(B)']),
    precision: num(m['metrics/precision(B)']),
    recall: num(m['metrics/recall(B)']),
    train_box: num(m['train/box_loss']),
    train_cls: num(m['train/cls_loss']),
    train_dfl: num(m['train/dfl_loss']),
    val_box: num(m['val/box_loss']),
    val_cls: num(m['val/cls_loss']),
    val_dfl: num(m['val/dfl_loss']),
    lr: num(m['lr/pg0']),
  }
}

function formatDuration(seconds?: number): string {
  if (!seconds || seconds < 0) return '–'
  const s = Math.round(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${sec}s`
  return `${sec}s`
}

const PLOT_LABELS: { match: RegExp; label: string }[] = [
  { match: /^results\.png$/i, label: 'Training curves' },
  { match: /confusion_matrix_normalized/i, label: 'Confusion matrix (norm.)' },
  { match: /confusion_matrix/i, label: 'Confusion matrix' },
  { match: /PR_curve/i, label: 'Precision–Recall' },
  { match: /P_curve/i, label: 'Precision' },
  { match: /R_curve/i, label: 'Recall' },
  { match: /F1_curve/i, label: 'F1' },
  { match: /labels_correlogram/i, label: 'Label correlogram' },
  { match: /labels\.jpg/i, label: 'Label distribution' },
]

export default function TrainRunDetailPage() {
  const { projectId = '', runId = '' } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const run = useQuery({
    queryKey: ['train-run', runId],
    queryFn: () => api.get<TrainRunOut>(`/training/runs/${runId}`),
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
      if (ev.phase !== 'epoch') {
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
    livePhase === 'start' ? 'Starting…' : livePhase === 'preparing' ? 'Preparing…' : 'Initializing…'

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
                    Loading model & scanning the dataset — the first epoch will start shortly.
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
                      {r.params.batch}
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

function ChartCard({
  title,
  hint,
  children,
}: {
  title: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <Card withBorder radius="md" padding="md">
      <Group justify="space-between" mb="xs">
        <Text size="sm" fw={600}>
          {title}
        </Text>
        {hint && (
          <Text size="xs" c="dimmed">
            {hint}
          </Text>
        )}
      </Group>
      {children}
    </Card>
  )
}

function PlotsSection({
  files,
  onOpen,
}: {
  files: { name: string; url: string }[]
  onOpen: (v: { url: string; label: string }) => void
}) {
  const images = files.filter((f) => /\.(png|jpe?g)$/i.test(f.name))
  const key: { file: { name: string; url: string }; label: string }[] = []
  const preds: { name: string; url: string }[] = []
  for (const f of images) {
    if (/val_batch\d+_pred/i.test(f.name)) {
      preds.push(f)
      continue
    }
    const hit = PLOT_LABELS.find((p) => p.match.test(f.name))
    if (hit) key.push({ file: f, label: hit.label })
  }
  if (!key.length && !preds.length) return null
  return (
    <Stack gap="xs">
      {!!key.length && (
        <>
          <Text fw={600} size="sm">
            Results
          </Text>
          <SimpleGrid cols={{ base: 2, md: 4 }}>
            {key.map(({ file, label }) => (
              <Card
                key={file.name}
                withBorder
                p="xs"
                radius="md"
                style={{ cursor: 'pointer' }}
                onClick={() => onOpen({ url: file.url, label })}
              >
                <Image src={file.url} radius="sm" loading="lazy" h={140} fit="contain" />
                <Text size="xs" fw={500} mt={4} ta="center">
                  {label}
                </Text>
              </Card>
            ))}
          </SimpleGrid>
        </>
      )}
      {!!preds.length && (
        <>
          <Text fw={600} size="sm" mt="xs">
            Prediction samples
          </Text>
          <SimpleGrid cols={{ base: 2, md: 3 }}>
            {preds.map((f) => (
              <Card
                key={f.name}
                withBorder
                p="xs"
                radius="md"
                style={{ cursor: 'pointer' }}
                onClick={() => onOpen({ url: f.url, label: f.name })}
              >
                <Image src={f.url} radius="sm" loading="lazy" />
              </Card>
            ))}
          </SimpleGrid>
        </>
      )}
    </Stack>
  )
}

type PcMetric = 'mAP50-95' | 'mAP50' | 'precision' | 'recall'
const PC_METRICS: { value: PcMetric; label: string }[] = [
  { value: 'mAP50-95', label: 'mAP50-95' },
  { value: 'mAP50', label: 'mAP50' },
  { value: 'precision', label: 'P' },
  { value: 'recall', label: 'R' },
]

function PerClassEpochChart({ history }: { history: PerClassEpoch[] }) {
  const [metric, setMetric] = useState<PcMetric>('mAP50-95')
  const [picked, setPicked] = useState<string[]>([])

  // class identity/order from the last epoch's snapshot
  const classes = history[history.length - 1]?.metrics ?? []

  const shown = useMemo(() => {
    if (picked.length) return classes.filter((c) => picked.includes(String(c.cls)))
    if (classes.length > 8) {
      return [...classes].sort((a, b) => (b[metric] as number) - (a[metric] as number)).slice(0, 8)
    }
    return classes
  }, [classes, picked, metric])

  const data = useMemo(
    () =>
      history.map((h) => {
        const row: Record<string, number> = { epoch: h.epoch }
        for (const c of shown) {
          const m = h.metrics.find((x) => x.cls === c.cls)
          if (m) row[c.name] = m[metric] as number
        }
        return row
      }),
    [history, shown, metric],
  )

  const series = shown.map((c) => ({ name: c.name, color: classColor(c.cls) }))

  return (
    <Card withBorder radius="md" padding="md">
      <Group justify="space-between" wrap="wrap" mb="xs" gap="xs">
        <Text size="sm" fw={600}>
          Per-class over epochs
        </Text>
        <Group gap="xs">
          <SegmentedControl
            size="xs"
            value={metric}
            onChange={(v) => setMetric(v as PcMetric)}
            data={PC_METRICS}
          />
          <MultiSelect
            size="xs"
            w={220}
            placeholder={
              picked.length ? undefined : classes.length > 8 ? 'Top 8 (pick classes)' : 'All classes'
            }
            data={classes.map((c) => ({ value: String(c.cls), label: c.name }))}
            value={picked}
            onChange={setPicked}
            clearable
            searchable
            maxDropdownHeight={220}
          />
        </Group>
      </Group>
      <LineChart
        h={240}
        data={data}
        dataKey="epoch"
        series={series}
        curveType="monotone"
        withDots={data.length < 40}
        withLegend
        yAxisProps={{ domain: [0, 1] }}
        valueFormatter={(v) => v.toFixed(3)}
      />
    </Card>
  )
}

type PcKey = 'name' | 'instances' | 'precision' | 'recall' | 'mAP50' | 'mAP50-95'

function mapColor(v: number): string {
  if (v < 0.3) return 'red'
  if (v < 0.5) return 'orange'
  if (v < 0.7) return 'yellow'
  return 'teal'
}

function PerClassTable({ rows }: { rows: PerClassRow[] }) {
  const [sortKey, setSortKey] = useState<PcKey>('mAP50-95')
  const [asc, setAsc] = useState(true) // weak classes first by default
  const hasInstances = rows.some((r) => r.instances != null)

  const sorted = [...rows].sort((a, b) => {
    const av = a[sortKey]
    const bv = b[sortKey]
    if (typeof av === 'string' || typeof bv === 'string') {
      return (asc ? 1 : -1) * String(av).localeCompare(String(bv))
    }
    return (asc ? 1 : -1) * ((av ?? -Infinity) - (bv ?? -Infinity))
  })

  const toggle = (k: PcKey) => {
    if (k === sortKey) setAsc((v) => !v)
    else {
      setSortKey(k)
      setAsc(true)
    }
  }

  const Th = ({ k, children, ta }: { k: PcKey; children: React.ReactNode; ta?: 'right' }) => (
    <Table.Th
      style={{ cursor: 'pointer', textAlign: ta }}
      onClick={() => toggle(k)}
    >
      {children}
      {sortKey === k ? (asc ? ' ▲' : ' ▼') : ''}
    </Table.Th>
  )

  return (
    <Card withBorder radius="md" padding="md">
      <Text size="sm" fw={600} mb="xs">
        Per-class metrics
      </Text>
      <Table.ScrollContainer minWidth={480}>
        <Table highlightOnHover verticalSpacing={6} horizontalSpacing="md">
          <Table.Thead>
            <Table.Tr>
              <Th k="name">Class</Th>
              {hasInstances && <Th k="instances" ta="right">Instances</Th>}
              <Th k="precision" ta="right">P</Th>
              <Th k="recall" ta="right">R</Th>
              <Th k="mAP50" ta="right">mAP50</Th>
              <Th k="mAP50-95" ta="right">mAP50-95</Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.map((r) => (
              <Table.Tr key={r.cls}>
                <Table.Td>
                  <Group gap={6} wrap="nowrap">
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 3,
                        background: classColor(r.cls),
                        flexShrink: 0,
                      }}
                    />
                    <Text size="sm" fw={500} truncate>
                      {r.name}
                    </Text>
                  </Group>
                </Table.Td>
                {hasInstances && (
                  <Table.Td ta="right">
                    <Text size="sm" c="dimmed">
                      {r.instances ?? '–'}
                    </Text>
                  </Table.Td>
                )}
                <Table.Td ta="right">{r.precision.toFixed(3)}</Table.Td>
                <Table.Td ta="right">{r.recall.toFixed(3)}</Table.Td>
                <Table.Td ta="right">{r.mAP50.toFixed(3)}</Table.Td>
                <Table.Td>
                  <Group gap="xs" wrap="nowrap" justify="flex-end">
                    <Progress
                      value={r['mAP50-95'] * 100}
                      color={mapColor(r['mAP50-95'])}
                      w={64}
                      size="sm"
                    />
                    <Text size="sm" fw={600} w={44} ta="right">
                      {r['mAP50-95'].toFixed(3)}
                    </Text>
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Card>
  )
}

// primary params stay visible; everything else hides behind the toggle
const PRIMARY_PARAMS = ['epochs', 'imgsz', 'batch']

function DetailsCard({ run, durationSec }: { run: TrainRunOut; durationSec?: number }) {
  const [showParams, setShowParams] = useState(false)
  const metaRows: [string, React.ReactNode][] = [
    ['Base model', run.base_model_name ?? run.base_model_id],
    ['Status', run.status],
    ['Started', new Date(run.created_at).toLocaleString()],
    ['Finished', run.finished_at ? new Date(run.finished_at).toLocaleString() : '–'],
    ['Duration', formatDuration(durationSec)],
    ...PRIMARY_PARAMS.filter((k) => k in run.params).map(
      (k) => [k, String(run.params[k])] as [string, React.ReactNode],
    ),
  ]
  const paramRows = Object.entries(run.params)
    .filter(([k]) => !PRIMARY_PARAMS.includes(k))
    .map(([k, v]) => [k, String(v)] as [string, React.ReactNode])
  const renderRows = (rows: [string, React.ReactNode][]) =>
    rows.map(([k, v]) => (
      <Table.Tr key={k}>
        <Table.Th w={160} style={{ color: 'var(--mantine-color-dimmed)', fontWeight: 500 }}>
          {k}
        </Table.Th>
        <Table.Td>
          <Text size="sm">{v}</Text>
        </Table.Td>
      </Table.Tr>
    ))
  return (
    <Card withBorder radius="md" padding="md">
      <Text size="sm" fw={600} mb="xs">
        Run details
      </Text>
      <Table variant="vertical" withRowBorders={false} verticalSpacing={4}>
        <Table.Tbody>{renderRows(metaRows)}</Table.Tbody>
      </Table>
      {paramRows.length > 0 && (
        <>
          <Button
            variant="subtle"
            size="compact-sm"
            mt={4}
            rightSection={<IconChevronDown size={14} />}
            onClick={() => setShowParams((v) => !v)}
          >
            {showParams ? 'Hide' : 'Show'} parameters ({paramRows.length})
          </Button>
          <Collapse expanded={showParams}>
            <Table variant="vertical" withRowBorders={false} verticalSpacing={4}>
              <Table.Tbody>{renderRows(paramRows)}</Table.Tbody>
            </Table>
          </Collapse>
        </>
      )}
    </Card>
  )
}

import { useEffect, useRef, useState } from 'react'
import {
  ActionIcon,
  Anchor,
  Badge,
  Button,
  Card,
  Group,
  Image,
  Progress,
  RingProgress,
  SimpleGrid,
  Stack,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { LineChart } from '@mantine/charts'
import { IconArrowLeft, IconDownload, IconPlaylistAdd } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import {
  api,
  subscribeTrainEvents,
  type ModelOut,
  type TrainEpochEvent,
  type TrainRunOut,
} from '../api/client'
const RUN_STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  stopped: 'yellow',
  queued: 'gray',
}

interface EpochPoint {
  epoch: number
  mAP50?: number
  'mAP50-95'?: number
  precision?: number
  recall?: number
  box_loss?: number
  cls_loss?: number
  dfl_loss?: number
  [key: string]: number | undefined
}

function toPoint(ev: TrainEpochEvent): EpochPoint | null {
  if (ev.phase !== 'epoch' || !ev.epoch || !ev.metrics) return null
  if (ev.epochs && ev.epoch > ev.epochs) return null // final extra val pass
  const m = ev.metrics
  return {
    epoch: ev.epoch,
    mAP50: m['metrics/mAP50(B)'],
    'mAP50-95': m['metrics/mAP50-95(B)'],
    precision: m['metrics/precision(B)'],
    recall: m['metrics/recall(B)'],
    box_loss: m['val/box_loss'],
    cls_loss: m['val/cls_loss'],
    dfl_loss: m['val/dfl_loss'],
  }
}

export default function TrainRunDetailPage() {
  const { projectId = '', runId = '' } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const run = useQuery({
    queryKey: ['train-run', runId],
    queryFn: () => api.get<TrainRunOut>(`/train/runs/${runId}`),
  })

  const [points, setPoints] = useState<EpochPoint[]>([])
  const [liveStatus, setLiveStatus] = useState<TrainEpochEvent | null>(null)
  const unsubscribe = useRef<(() => void) | null>(null)

  useEffect(() => {
    setPoints([])
    setLiveStatus(null)
    api.get<TrainEpochEvent[]>(`/train/runs/${runId}/history`).then((history) => {
      setPoints(history.map(toPoint).filter(Boolean) as EpochPoint[])
      const last = history[history.length - 1]
      if (last) setLiveStatus(last)
    })
    unsubscribe.current?.()
    unsubscribe.current = subscribeTrainEvents(runId, (ev) => {
      setLiveStatus(ev)
      const p = toPoint(ev)
      if (p) setPoints((prev) => (prev.some((x) => x.epoch === p.epoch) ? prev : [...prev, p]))
      if (ev.phase !== 'epoch') {
        queryClient.invalidateQueries({ queryKey: ['train-run', runId] })
        queryClient.invalidateQueries({ queryKey: ['train-runs', projectId] })
        queryClient.invalidateQueries({ queryKey: ['train-artifacts', runId] })
      }
    })
    return () => unsubscribe.current?.()
  }, [runId]) // eslint-disable-line react-hooks/exhaustive-deps

  const artifacts = useQuery({
    queryKey: ['train-artifacts', runId],
    queryFn: () =>
      api.get<{ files: { name: string; url: string }[]; weights: { name: string; url: string }[] }>(
        `/train/runs/${runId}/artifacts`,
      ),
  })

  const stop = useMutation({
    mutationFn: () => api.post<TrainRunOut>(`/train/runs/${runId}/stop`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['train-run', runId] })
      queryClient.invalidateQueries({ queryKey: ['train-runs', projectId] })
    },
  })

  const register = useMutation({
    mutationFn: (which: string) => api.post<ModelOut>(`/train/runs/${runId}/register`, { which }),
    onSuccess: (m) => {
      notifications.show({ message: `Added to models: ${m.name}`, color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['models', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const r = run.data
  const epochTotal =
    (liveStatus?.phase === 'epoch' ? liveStatus.epochs : undefined) ??
    Number(r?.params.epochs ?? 0)
  const rawEpoch = liveStatus?.phase === 'epoch' ? (liveStatus.epoch ?? 0) : points.length
  // ultralytics emits one extra val pass after the last epoch — clamp for display
  const epochNow = epochTotal ? Math.min(rawEpoch, epochTotal) : rawEpoch
  const progressPct =
    r?.status === 'done'
      ? 100
      : epochTotal
        ? Math.min(100, Math.round((epochNow / epochTotal) * 100))
        : 0
  const latest = points[points.length - 1]

  return (
    <Stack gap="md">
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
          {r?.status === 'running' && (
            <Button size="xs" color="red" variant="light" onClick={() => stop.mutate()}>
              Stop
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

      {r && (
        <Card withBorder radius="md" padding="md">
          <Group justify="space-between" wrap="wrap">
            <Group gap="lg">
              <RingProgress
                size={84}
                thickness={8}
                roundCaps
                sections={[{ value: progressPct, color: r.status === 'error' ? 'red' : 'blue' }]}
                label={
                  <Text size="xs" ta="center" fw={700}>
                    {progressPct}%
                  </Text>
                }
              />
              <div>
                <Text size="sm" fw={600}>
                  epoch {epochNow || '–'} / {epochTotal || '–'}
                </Text>
                <Text size="xs" c="dimmed">
                  {r.base_model_name ?? r.base_model_id} · ep {r.params.epochs} · {r.params.imgsz}
                  px · batch {r.params.batch}
                </Text>
                <Text size="xs" c="dimmed">
                  Started {new Date(r.created_at).toLocaleString()}
                  {r.finished_at ? ` · finished ${new Date(r.finished_at).toLocaleString()}` : ''}
                </Text>
              </div>
            </Group>
            <Group gap="xs">
              {latest?.mAP50 != null && (
                <Badge variant="light" color="teal" size="lg">
                  mAP50 {latest.mAP50.toFixed(3)}
                </Badge>
              )}
              {latest?.['mAP50-95'] != null && (
                <Badge variant="light" color="blue" size="lg">
                  mAP50-95 {latest['mAP50-95'].toFixed(3)}
                </Badge>
              )}
              {latest?.box_loss != null && (
                <Badge variant="light" color="orange" size="lg">
                  box_loss {latest.box_loss.toFixed(3)}
                </Badge>
              )}
            </Group>
          </Group>
          {r.status === 'running' && <Progress value={progressPct} animated mt="sm" />}
        </Card>
      )}

      {points.length > 0 && (
        <SimpleGrid cols={{ base: 1, md: 3 }}>
          <Card withBorder radius="md">
            <Text size="sm" fw={600} mb="xs">
              mAP
            </Text>
            <LineChart
              h={200}
              data={points}
              dataKey="epoch"
              series={[
                { name: 'mAP50', color: 'teal.6' },
                { name: 'mAP50-95', color: 'blue.6' },
              ]}
              curveType="monotone"
              withDots={points.length < 30}
              yAxisProps={{ domain: [0, 1] }}
            />
          </Card>
          <Card withBorder radius="md">
            <Text size="sm" fw={600} mb="xs">
              Validation Loss
            </Text>
            <LineChart
              h={200}
              data={points}
              dataKey="epoch"
              series={[
                { name: 'box_loss', color: 'orange.6' },
                { name: 'cls_loss', color: 'red.6' },
                { name: 'dfl_loss', color: 'grape.6' },
              ]}
              curveType="monotone"
              withDots={points.length < 30}
            />
          </Card>
          <Card withBorder radius="md">
            <Text size="sm" fw={600} mb="xs">
              Precision / Recall
            </Text>
            <LineChart
              h={200}
              data={points}
              dataKey="epoch"
              series={[
                { name: 'precision', color: 'cyan.6' },
                { name: 'recall', color: 'indigo.6' },
              ]}
              curveType="monotone"
              withDots={points.length < 30}
              yAxisProps={{ domain: [0, 1] }}
            />
          </Card>
        </SimpleGrid>
      )}

      {r?.metrics && (
        <Group gap="xs">
          {Object.entries(r.metrics)
            .filter(([k]) => k.startsWith('metrics/') || k === 'fitness')
            .map(([k, v]) => (
              <Badge key={k} variant="light" size="lg">
                {k.replace('metrics/', '')}: {v.toFixed(3)}
              </Badge>
            ))}
        </Group>
      )}

      {!!artifacts.data?.files.length && (
        <>
          <Text fw={600} size="sm">
            Artifacts
          </Text>
          <SimpleGrid cols={{ base: 2, md: 3 }}>
            {artifacts.data.files
              .filter((f) => f.name.endsWith('.png') || f.name.endsWith('.jpg'))
              .map((f) => (
                <Card key={f.name} withBorder p="xs" radius="md">
                  <Anchor href={f.url} target="_blank">
                    <Image src={f.url} radius="sm" loading="lazy" />
                  </Anchor>
                  <Text size="xs" c="dimmed" truncate>
                    {f.name}
                  </Text>
                </Card>
              ))}
          </SimpleGrid>
        </>
      )}
    </Stack>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Anchor,
  Badge,
  Button,
  Card,
  Group,
  Image,
  NumberInput,
  Progress,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { LineChart } from '@mantine/charts'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  subscribeTrainEvents,
  type ModelOut,
  type TrainDataset,
  type TrainEpochEvent,
  type TrainRunOut,
} from '../api/client'

const STATUS_COLOR: Record<string, string> = {
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
  box_loss?: number
  cls_loss?: number
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
    box_loss: m['val/box_loss'],
    cls_loss: m['val/cls_loss'],
  }
}

export default function TrainingPage() {
  const queryClient = useQueryClient()
  const [selectedRun, setSelectedRun] = useState<string | null>(null)

  // launch form state
  const [datasetKey, setDatasetKey] = useState<string | null>(null)
  const [baseModel, setBaseModel] = useState<string | null>(null)
  const [runName, setRunName] = useState('')
  const [epochs, setEpochs] = useState<number | string>(100)
  const [imgsz, setImgsz] = useState<number | string>(640)
  const [batch, setBatch] = useState<number | string>(16)
  const [device, setDevice] = useState<string | null>('auto')
  const [advanced, setAdvanced] = useState(false)
  const [patience, setPatience] = useState<number | string>(100)
  const [lr0, setLr0] = useState<number | string>('')
  const [optimizer, setOptimizer] = useState<string | null>(null)

  const datasets = useQuery({
    queryKey: ['train-datasets'],
    queryFn: () => api.get<TrainDataset[]>('/train/datasets'),
  })
  const models = useQuery({
    queryKey: ['models'],
    queryFn: () => api.get<ModelOut[]>('/models'),
  })
  const runs = useQuery({
    queryKey: ['train-runs'],
    queryFn: () => api.get<TrainRunOut[]>('/train/runs'),
  })

  const run = runs.data?.find((r) => r.id === selectedRun) ?? null

  // chart data: history + live SSE
  const [points, setPoints] = useState<EpochPoint[]>([])
  const [liveStatus, setLiveStatus] = useState<TrainEpochEvent | null>(null)
  const unsubscribe = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (!selectedRun) return
    setPoints([])
    setLiveStatus(null)
    api.get<TrainEpochEvent[]>(`/train/runs/${selectedRun}/history`).then((history) => {
      setPoints(history.map(toPoint).filter(Boolean) as EpochPoint[])
      const last = history[history.length - 1]
      if (last) setLiveStatus(last)
    })
    unsubscribe.current?.()
    unsubscribe.current = subscribeTrainEvents(selectedRun, (ev) => {
      setLiveStatus(ev)
      const p = toPoint(ev)
      if (p) setPoints((prev) => (prev.some((x) => x.epoch === p.epoch) ? prev : [...prev, p]))
      if (ev.phase !== 'epoch') {
        queryClient.invalidateQueries({ queryKey: ['train-runs'] })
        queryClient.invalidateQueries({ queryKey: ['train-artifacts', selectedRun] })
      }
    })
    return () => unsubscribe.current?.()
  }, [selectedRun]) // eslint-disable-line react-hooks/exhaustive-deps

  const artifacts = useQuery({
    queryKey: ['train-artifacts', selectedRun],
    queryFn: () =>
      api.get<{ files: { name: string; url: string }[]; weights: { name: string; url: string }[] }>(
        `/train/runs/${selectedRun}/artifacts`,
      ),
    enabled: !!selectedRun,
  })

  const launch = useMutation({
    mutationFn: () => {
      const [projectId, exportId] = (datasetKey ?? '').split('|')
      return api.post<TrainRunOut>('/train/runs', {
        name: runName.trim() || null,
        project_id: projectId,
        export_id: exportId,
        base_model_id: baseModel,
        device: device === 'auto' ? null : device,
        params: {
          epochs: Number(epochs),
          imgsz: Number(imgsz),
          batch: Number(batch),
          patience: Number(patience),
          ...(lr0 !== '' ? { lr0: Number(lr0) } : {}),
          ...(optimizer ? { optimizer } : {}),
        },
      })
    },
    onSuccess: (r) => {
      notifications.show({ message: `학습 시작: ${r.name}`, color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['train-runs'] })
      setSelectedRun(r.id)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const stop = useMutation({
    mutationFn: () => api.post<TrainRunOut>(`/train/runs/${selectedRun}/stop`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['train-runs'] }),
  })

  const register = useMutation({
    mutationFn: (which: string) =>
      api.post<ModelOut>(`/train/runs/${selectedRun}/register`, { which }),
    onSuccess: (m) => {
      notifications.show({ message: `모델 레지스트리에 등록됨: ${m.name}`, color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const datasetOptions = useMemo(
    () =>
      datasets.data?.map((d) => ({
        value: `${d.project_id}|${d.export_id}`,
        label: `${d.project_name} / ${d.export_id} (train ${d.train} · val ${d.val} · ${d.classes}cls)`,
      })) ?? [],
    [datasets.data],
  )

  const progressPct =
    liveStatus?.phase === 'epoch' && liveStatus.epoch && liveStatus.epochs
      ? Math.min(100, Math.round((liveStatus.epoch / liveStatus.epochs) * 100))
      : run?.status === 'done'
        ? 100
        : 0

  return (
    <Stack>
      <Title order={3}>학습</Title>

      <Card withBorder>
        <Text fw={600} mb="xs">
          새 학습 실행
        </Text>
        <Stack gap="xs">
          <Group grow>
            <Select
              label="데이터셋 (프로젝트 내보내기)"
              placeholder={datasetOptions.length ? '선택' : '먼저 프로젝트에서 내보내기를 만드세요'}
              data={datasetOptions}
              value={datasetKey}
              onChange={setDatasetKey}
              searchable
            />
            <Select
              label="베이스 모델"
              placeholder="레지스트리에서 선택"
              data={models.data?.map((m) => ({ value: m.id, label: m.name })) ?? []}
              value={baseModel}
              onChange={setBaseModel}
              searchable
            />
          </Group>
          <Group grow>
            <TextInput
              label="런 이름 (선택)"
              value={runName}
              onChange={(e) => setRunName(e.currentTarget.value)}
            />
            <NumberInput label="Epochs" value={epochs} onChange={setEpochs} min={1} />
            <NumberInput label="이미지 크기" value={imgsz} onChange={setImgsz} min={64} step={32} />
            <NumberInput label="배치" value={batch} onChange={setBatch} min={1} />
            <Select
              label="디바이스"
              data={['auto', 'cpu', 'mps', '0']}
              value={device}
              onChange={setDevice}
            />
          </Group>
          <Anchor size="sm" onClick={() => setAdvanced(!advanced)}>
            고급 설정 {advanced ? '접기' : '펼치기'}
          </Anchor>
          {advanced && (
            <Group grow>
              <NumberInput label="Patience" value={patience} onChange={setPatience} min={0} />
              <NumberInput
                label="초기 학습률 (lr0)"
                value={lr0}
                onChange={setLr0}
                step={0.001}
                decimalScale={4}
              />
              <Select
                label="옵티마이저"
                placeholder="auto"
                data={['SGD', 'Adam', 'AdamW', 'RMSProp']}
                value={optimizer}
                onChange={setOptimizer}
                clearable
              />
            </Group>
          )}
          <Group justify="flex-end">
            <Button
              onClick={() => launch.mutate()}
              disabled={!datasetKey || !baseModel}
              loading={launch.isPending}
            >
              학습 시작
            </Button>
          </Group>
        </Stack>
      </Card>

      <Group align="flex-start" gap="md" wrap="nowrap">
        <Card withBorder w={360} style={{ flexShrink: 0 }}>
          <Text fw={600} mb="xs">
            학습 이력
          </Text>
          <Table highlightOnHover>
            <Table.Tbody>
              {runs.data?.map((r) => (
                <Table.Tr
                  key={r.id}
                  style={{
                    cursor: 'pointer',
                    background:
                      r.id === selectedRun ? 'var(--mantine-color-blue-light)' : undefined,
                  }}
                  onClick={() => setSelectedRun(r.id)}
                >
                  <Table.Td>
                    <Text size="sm" fw={500}>
                      {r.name}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {r.base_model_name} · {new Date(r.created_at).toLocaleString()}
                    </Text>
                  </Table.Td>
                  <Table.Td w={80} align="right">
                    <Badge color={STATUS_COLOR[r.status] ?? 'gray'} variant="light">
                      {r.status}
                    </Badge>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
          {runs.data?.length === 0 && (
            <Text size="sm" c="dimmed">
              아직 학습 이력이 없습니다.
            </Text>
          )}
        </Card>

        <Stack style={{ flex: 1, minWidth: 0 }}>
          {!run && <Text c="dimmed">좌측에서 학습 런을 선택하세요.</Text>}
          {run && (
            <>
              <Group justify="space-between">
                <Group gap="xs">
                  <Text fw={600}>{run.name}</Text>
                  <Badge color={STATUS_COLOR[run.status] ?? 'gray'} variant="light">
                    {run.status}
                  </Badge>
                  {run.error && (
                    <Text size="sm" c="red">
                      {run.error}
                    </Text>
                  )}
                </Group>
                <Group gap="xs">
                  {run.status === 'running' && (
                    <Button size="xs" color="red" variant="light" onClick={() => stop.mutate()}>
                      중단
                    </Button>
                  )}
                  {artifacts.data?.weights.map((w) => (
                    <Anchor key={w.name} href={w.url} download size="sm">
                      {w.name} 다운로드
                    </Anchor>
                  ))}
                  {(run.status === 'done' || run.status === 'stopped') && (
                    <Button
                      size="xs"
                      variant="light"
                      onClick={() => register.mutate(run.status === 'done' ? 'best' : 'last')}
                      loading={register.isPending}
                    >
                      레지스트리에 등록
                    </Button>
                  )}
                </Group>
              </Group>

              {run.status === 'running' && (
                <Stack gap={4}>
                  <Progress value={progressPct} animated />
                  <Text size="xs" c="dimmed">
                    epoch {liveStatus?.phase === 'epoch' ? liveStatus.epoch : '-'} /{' '}
                    {liveStatus?.phase === 'epoch' ? liveStatus.epochs : run.params.epochs}
                  </Text>
                </Stack>
              )}

              {points.length > 0 && (
                <SimpleGrid cols={{ base: 1, md: 2 }}>
                  <Card withBorder>
                    <Text size="sm" fw={600} mb="xs">
                      mAP
                    </Text>
                    <LineChart
                      h={220}
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
                  <Card withBorder>
                    <Text size="sm" fw={600} mb="xs">
                      Validation Loss
                    </Text>
                    <LineChart
                      h={220}
                      data={points}
                      dataKey="epoch"
                      series={[
                        { name: 'box_loss', color: 'orange.6' },
                        { name: 'cls_loss', color: 'red.6' },
                      ]}
                      curveType="monotone"
                      withDots={points.length < 30}
                    />
                  </Card>
                </SimpleGrid>
              )}

              {run.metrics && (
                <Group gap="xs">
                  {Object.entries(run.metrics)
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
                    결과 아티팩트
                  </Text>
                  <SimpleGrid cols={{ base: 2, md: 3 }}>
                    {artifacts.data.files
                      .filter((f) => f.name.endsWith('.png') || f.name.endsWith('.jpg'))
                      .map((f) => (
                        <Card key={f.name} withBorder p="xs">
                          <Image src={f.url} radius="sm" loading="lazy" />
                          <Text size="xs" c="dimmed" truncate>
                            {f.name}
                          </Text>
                        </Card>
                      ))}
                  </SimpleGrid>
                </>
              )}
            </>
          )}
        </Stack>
      </Group>
    </Stack>
  )
}

import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Checkbox,
  Group,
  NumberInput,
  Select,
  SimpleGrid,
  Slider,
  Stack,
  Switch,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { IconAlertTriangle, IconPlayerPlay } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import {
  api,
  DEFAULT_TILING,
  getResources,
  listDatasets,
  previewTiling,
  type ModelOut,
  type SplitPreview,
  type TilingParams,
  type TilingPreview,
  type TrainRunOut,
} from '../api/client'

// 미리보기는 스캔 결과(positive/hard/incidental)만 서버에서 받는다 — 비율·keep_all_negatives 는
// 스캔에 영향을 주지 않으므로 재요청 없이 여기서 백엔드와 같은 규칙으로 다시 계산한다.
// (하드 네거티브를 전부 채우고 나서, 인시덴탈로 목표까지 채운다)
function computeActual(s: SplitPreview, tiling: TilingParams) {
  const target = Math.round(s.positive * tiling.negative_ratio)
  const keptIncidental = tiling.keep_all_negatives
    ? s.incidental
    : Math.min(s.incidental, Math.max(0, target - s.hard))
  const negativeKept = s.hard + keptIncidental
  const total = s.positive + negativeKept
  const overshoot = !tiling.keep_all_negatives && s.hard > target
  return { negativeKept, total, overshoot }
}

export default function TrainPage() {
  const { projectId = '' } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [datasetToken, setDatasetToken] = useState<string | null>(null)
  const [baseModel, setBaseModel] = useState<string | null>(null)
  const [runName, setRunName] = useState('')
  const [epochs, setEpochs] = useState<number | string>(100)
  const [imgsz, setImgsz] = useState<number | string>(640)
  const [batch, setBatch] = useState<number | string>(16)
  const [advanced, setAdvanced] = useState(false)
  const [patience, setPatience] = useState<number | string>(100)
  // ultralytics default lr0 is 0.01 — prefill it so the default is visible
  const [lr0, setLr0] = useState<number | string>(0.01)
  // 'auto' lets ultralytics pick the optimizer AND override lr0 (so lr0 is
  // disabled then); picking a concrete optimizer prefills its sensible lr0.
  const [optimizer, setOptimizer] = useState<string>('auto')
  // augmentation — prefilled with ultralytics defaults so they're visible
  const [fliplr, setFliplr] = useState<number | string>(0.5)
  const [flipud, setFlipud] = useState<number | string>(0)
  const [degrees, setDegrees] = useState<number | string>(0)
  const [scale, setScale] = useState<number | string>(0.5)
  const [mosaic, setMosaic] = useState<number | string>(1.0)
  const [mixup, setMixup] = useState<number | string>(0)

  const [tiling, setTiling] = useState<TilingParams>(DEFAULT_TILING)
  const [preview, setPreview] = useState<TilingPreview | null>(null)

  // 격자를 흔드는 노브가 바뀌면 미리보기는 더 이상 이 설정의 것이 아니다
  useEffect(() => {
    setPreview(null)
  }, [tiling.tile_size, tiling.stride, tiling.min_visibility, datasetToken])

  const previewRun = useMutation({
    mutationFn: () => previewTiling(datasetToken ?? '', tiling),
    onSuccess: setPreview,
    onError: (e) => {
      setPreview(null)
      notifications.show({ message: String(e), color: 'red' })
    },
  })

  const resources = useQuery({
    queryKey: ['resources'],
    queryFn: getResources,
    refetchInterval: 5000,
  })
  const trainingActive = resources.data?.training_active ?? false

  const datasets = useQuery({
    queryKey: ['datasets', projectId],
    queryFn: () => listDatasets(projectId),
  })
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })

  // 학습에 쓸 수 있는 것은 **train 이 있는** 데이터셋뿐이다 — 검수하고 나눠야 생긴다
  const datasetOptions = useMemo(
    () =>
      (datasets.data ?? [])
        .filter((d) => d.train > 0)
        .map((d) => ({
          value: `dataset:${projectId}:${d.id}`,
          label: `${d.name} (train ${d.train} · val ${d.val})`,
        })),
    [datasets.data, projectId],
  )

  const selected = datasets.data?.find((d) => `dataset:${projectId}:${d.id}` === datasetToken)

  // 시작하면 그 런의 상세로 넘어간다 — 진행률은 거기가 보여준다
  const launch = useMutation({
    mutationFn: () =>
      api.post<TrainRunOut>('/training/runs', {
        name: runName.trim() || null,
        project_id: projectId,
        dataset: datasetToken,
        base_model_id: baseModel,
        device: null, // backend resolves the configured device (auto)
        params: {
          epochs: Number(epochs),
          imgsz: Number(imgsz),
          batch: Number(batch),
          patience: Number(patience),
          // 'auto' overrides lr0 in ultralytics, so send neither; a concrete
          // optimizer sends both so the recorded params are truthful.
          ...(optimizer !== 'auto'
            ? { optimizer, ...(lr0 !== '' ? { lr0: Number(lr0) } : {}) }
            : {}),
          ...(fliplr !== '' ? { fliplr: Number(fliplr) } : {}),
          ...(flipud !== '' ? { flipud: Number(flipud) } : {}),
          ...(degrees !== '' ? { degrees: Number(degrees) } : {}),
          ...(scale !== '' ? { scale: Number(scale) } : {}),
          ...(mosaic !== '' ? { mosaic: Number(mosaic) } : {}),
          ...(mixup !== '' ? { mixup: Number(mixup) } : {}),
        },
        tiling,
      }),
    onSuccess: (r) => {
      notifications.show({ message: `Training started: ${r.name}`, color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['train-runs', projectId] })
      navigate(`/projects/${projectId}/training/${r.id}`)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  return (
    <Stack gap="lg">
      <div>
        <Title order={3}>Train</Title>
      </div>

      {trainingActive && (
        <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="Training in progress">
          A training run is already active. Starting another makes them compete for the GPU — both
          may slow down or fail.
        </Alert>
      )}

      <Card withBorder radius="md" padding="lg">
        <Stack gap="md">
          <div>
            <Text fw={600} mb={4}>
              1. Dataset
            </Text>
            <Stack gap="xs">
              {/* 학습은 데이터셋을 **직접** 먹는다 — 내보내기를 먼저 할 필요가 없다.
                  그 데이터셋의 train/val 이 그대로 학습에 쓰인다. */}
              <Select
                w={480}
                placeholder={
                  datasetOptions.length
                    ? 'Select a dataset'
                    : 'No datasets with a split yet — review and split some images first'
                }
                data={datasetOptions}
                value={datasetToken}
                onChange={setDatasetToken}
                disabled={!datasetOptions.length}
              />
              {selected && (
                <Text size="xs" c="dimmed">
                  train {selected.train} · val {selected.val} — test {selected.test} is held back
                </Text>
              )}
            </Stack>
          </div>

          <div>
            <Text fw={600} mb={4}>
              2. Base model
            </Text>
            <Select
              w={420}
              placeholder="Select from the model registry"
              data={models.data?.map((m) => ({ value: m.id, label: m.name })) ?? []}
              value={baseModel}
              onChange={setBaseModel}
            />
          </div>

          <div>
            <Text fw={600} mb={4}>
              3. Training settings
            </Text>
            <Group grow>
              <TextInput
                label="Run name (optional)"
                value={runName}
                onChange={(e) => setRunName(e.currentTarget.value)}
              />
              <NumberInput label="Epochs" value={epochs} onChange={setEpochs} min={1} />
              <NumberInput label="Image size" value={imgsz} onChange={setImgsz} min={64} step={32} />
              <NumberInput label="Batch" value={batch} onChange={setBatch} min={1} />
            </Group>

            <Stack gap="sm">
              <Switch
                label="Tile into patches before training"
                description="Cuts each frame into overlapping tiles so small objects survive downscaling. The test split is never tiled."
                checked={tiling.enabled}
                onChange={(e) => {
                  const enabled = e.currentTarget.checked
                  setTiling((t) => ({ ...t, enabled }))
                  // 타일 크기와 학습 imgsz 가 어긋나면 타일링의 이득이 사라진다
                  if (enabled) setImgsz(tiling.tile_size)
                }}
              />

              {tiling.enabled && (
                <>
                  <SimpleGrid cols={{ base: 1, sm: 3 }}>
                    <NumberInput
                      label="Tile size (px)"
                      value={tiling.tile_size}
                      onChange={(v) => {
                        const tile_size = Number(v) || 640
                        setTiling((t) => ({ ...t, tile_size }))
                        setImgsz(tile_size)
                      }}
                      min={64}
                      step={32}
                    />
                    <NumberInput
                      label="Stride (px)"
                      value={tiling.stride}
                      onChange={(v) => setTiling((t) => ({ ...t, stride: Number(v) || 480 }))}
                      min={32}
                      max={tiling.tile_size}
                      step={32}
                      error={
                        tiling.tile_size - tiling.stride <= 0
                          ? 'Stride must be < tile size — boundary objects split in two'
                          : undefined
                      }
                    />
                    <NumberInput
                      label="Min visibility (0–1)"
                      description="Boxes less visible than this are dropped, and so is their tile"
                      value={tiling.min_visibility}
                      onChange={(v) =>
                        setTiling((t) => ({
                          ...t,
                          min_visibility: v === '' || v == null ? 0.6 : Number(v),
                        }))
                      }
                      min={0}
                      max={1}
                      step={0.05}
                      decimalScale={2}
                    />
                  </SimpleGrid>

                  <Text size="xs" c="dimmed">
                    overlap {tiling.tile_size - tiling.stride}px
                  </Text>

                  <div>
                    <Group justify="space-between">
                      <Text size="sm">Negatives (% of positives, per split)</Text>
                      <Text size="sm" c="dimmed">
                        {tiling.keep_all_negatives
                          ? 'all'
                          : `${Math.round(tiling.negative_ratio * 100)}%`}
                      </Text>
                    </Group>
                    <Slider
                      value={Math.round(tiling.negative_ratio * 100)}
                      onChange={(v) => setTiling((t) => ({ ...t, negative_ratio: v / 100 }))}
                      min={0}
                      max={300}
                      step={5}
                      disabled={tiling.keep_all_negatives}
                      marks={[
                        { value: 0, label: '0' },
                        { value: 100, label: '1:1' },
                        { value: 300, label: '3:1' },
                      ]}
                      mb="md"
                    />
                    <Checkbox
                      label="Keep every negative tile"
                      checked={tiling.keep_all_negatives}
                      onChange={(e) =>
                        setTiling((t) => ({ ...t, keep_all_negatives: e.currentTarget.checked }))
                      }
                      size="xs"
                    />
                  </div>

                  <Group>
                    <Button
                      variant="light"
                      size="compact-sm"
                      loading={previewRun.isPending}
                      disabled={!datasetToken || tiling.tile_size - tiling.stride <= 0}
                      onClick={() => previewRun.mutate()}
                    >
                      Estimate
                    </Button>
                    {preview?.splits.map((s) => {
                      const { negativeKept, total, overshoot } = computeActual(s, tiling)
                      return (
                        <Text key={s.split} size="xs" c="dimmed">
                          <b>{s.split}</b> {s.positive} pos · {negativeKept} neg ({s.hard} hard +{' '}
                          {negativeKept - s.hard} incidental) = {total}
                          {overshoot ? ' ⚠ over target' : ''}
                        </Text>
                      )
                    })}
                  </Group>

                  {preview?.splits.some((s) => computeActual(s, tiling).overshoot) && (
                    <Alert color="yellow" variant="light" p="xs">
                      <Text size="xs">
                        Hard negatives (tiles from frames with no labels) alone exceed the target
                        ratio. They are never discarded — the actual ratio is higher than
                        requested.
                      </Text>
                    </Alert>
                  )}
                </>
              )}
            </Stack>

            <Anchor size="sm" onClick={() => setAdvanced(!advanced)} mt="xs" display="inline-block">
              {advanced ? 'Hide advanced settings' : 'Show advanced settings'}
            </Anchor>
            {advanced && (
              <Stack gap="md" mt="xs">
                <Group grow>
                  <NumberInput label="Patience" value={patience} onChange={setPatience} min={0} />
                  <NumberInput
                    label="Initial LR (lr0)"
                    value={lr0}
                    onChange={setLr0}
                    min={0}
                    step={0.001}
                    decimalScale={4}
                    disabled={optimizer === 'auto'}
                  />
                  <Select
                    label="Optimizer"
                    data={['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp']}
                    value={optimizer}
                    onChange={(v) => {
                      const opt = v ?? 'auto'
                      setOptimizer(opt)
                      // prefill each optimizer's sensible lr0 (SGD ~0.01, adaptive ~0.001)
                      if (opt !== 'auto') setLr0(opt === 'SGD' ? 0.01 : 0.001)
                    }}
                    allowDeselect={false}
                  />
                </Group>

                <div>
                  <Text fw={600} size="sm">
                    Augmentation
                  </Text>
                  <SimpleGrid cols={{ base: 2, sm: 3, md: 4 }} spacing="sm">
                    <NumberInput
                      label="Flip L/R"
                      value={fliplr}
                      onChange={setFliplr}
                      min={0}
                      max={1}
                      step={0.05}
                      decimalScale={2}
                    />
                    <NumberInput
                      label="Flip U/D"
                      value={flipud}
                      onChange={setFlipud}
                      min={0}
                      max={1}
                      step={0.05}
                      decimalScale={2}
                    />
                    <NumberInput
                      label="Rotation (°)"
                      value={degrees}
                      onChange={setDegrees}
                      min={0}
                      max={180}
                      step={5}
                    />
                    <NumberInput
                      label="Scale"
                      value={scale}
                      onChange={setScale}
                      min={0}
                      max={1}
                      step={0.05}
                      decimalScale={2}
                    />
                    <NumberInput
                      label="Mosaic"
                      value={mosaic}
                      onChange={setMosaic}
                      min={0}
                      max={1}
                      step={0.05}
                      decimalScale={2}
                    />
                    <NumberInput
                      label="Mixup"
                      value={mixup}
                      onChange={setMixup}
                      min={0}
                      max={1}
                      step={0.05}
                      decimalScale={2}
                    />
                  </SimpleGrid>
                </div>
              </Stack>
            )}
          </div>

          <Group justify="flex-end">
            {datasetToken && (
              <Badge variant="light" color="gray">
                {selected?.name}
              </Badge>
            )}
            <Button
              leftSection={<IconPlayerPlay size={16} />}
              onClick={() => launch.mutate()}
              disabled={!datasetToken || !baseModel}
              loading={launch.isPending}
            >
              Start training
            </Button>
          </Group>
        </Stack>
      </Card>
    </Stack>
  )
}

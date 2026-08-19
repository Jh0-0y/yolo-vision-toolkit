import { useMemo, useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Group,
  NumberInput,
  Select,
  SimpleGrid,
  Stack,
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
  getResources,
  listDatasets,
  type ModelOut,
  type TrainRunOut,
} from '../api/client'

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
        <Text c="dimmed" size="sm">
          Pick a dataset and base model — starting a run takes you to Training History.
        </Text>
      </div>

      {trainingActive && (
        <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="학습 진행 중">
          이미 학습이 진행 중입니다. 새 학습을 시작하면 GPU를 두고 경쟁해 둘 다 느려지거나 실패할 수
          있습니다.
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
                  <Text size="xs" c="dimmed" mb="xs">
                    Values are ultralytics defaults. Higher = stronger; 0 = off. Clear a field to
                    use the ultralytics default.
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

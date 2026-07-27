import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ActionIcon,
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
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconFileZip, IconPlayerPlay, IconTrash } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import {
  api,
  deleteTrainDataset,
  getResources,
  patchTrainDataset,
  type ModelOut,
  type TrainDataset,
  type TrainRunOut,
} from '../api/client'
import { useJobStore } from '../stores/jobStore'

export default function TrainPage() {
  const { projectId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [datasetToken, setDatasetToken] = useState<string | null>(null)
  const [baseModel, setBaseModel] = useState<string | null>(null)
  const [runName, setRunName] = useState('')
  const [epochs, setEpochs] = useState<number | string>(100)
  const [imgsz, setImgsz] = useState<number | string>(640)
  const [batch, setBatch] = useState<number | string>(16)
  const [device, setDevice] = useState<string | null>('auto')
  const [advanced, setAdvanced] = useState(false)
  const [patience, setPatience] = useState<number | string>(100)
  // ultralytics default lr0 is 0.01 — prefill it so the default is visible
  const [lr0, setLr0] = useState<number | string>(0.01)
  const [optimizer, setOptimizer] = useState<string | null>(null)
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
    queryKey: ['train-datasets', projectId],
    queryFn: () => api.get<TrainDataset[]>(`/training/datasets?project_id=${projectId}`),
  })
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })

  // dataset upload runs in a global job store so its progress survives page
  // navigation; the global JobIndicator renders the % bar and handles the toast.
  const startDatasetUpload = useJobStore((s) => s.startDatasetUpload)
  const jobsMap = useJobStore((s) => s.jobs)
  const datasetJobs = useMemo(
    () => Object.values(jobsMap).filter((j) => j.kind === 'dataset'),
    [jobsMap],
  )
  const uploadBusy = datasetJobs.some((j) => j.status === 'running')

  // auto-select a freshly uploaded dataset when its upload completes
  const pickedToken = useRef<string | null>(null)
  useEffect(() => {
    const doneJob = datasetJobs.find(
      (j) => j.status === 'done' && j.resultToken && j.resultToken !== pickedToken.current,
    )
    if (doneJob?.resultToken) {
      pickedToken.current = doneJob.resultToken
      setDatasetToken(doneJob.resultToken)
    }
  }, [datasetJobs])

  const removeDataset = useMutation({
    mutationFn: (datasetId: string) => deleteTrainDataset(datasetId),
    onSuccess: () => {
      notifications.show({ message: 'Dataset deleted', color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['train-datasets', projectId] })
      setDatasetToken(null)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const toggleAutoDelete = useMutation({
    mutationFn: ({ id, value }: { id: string; value: boolean }) =>
      patchTrainDataset(id, value),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['train-datasets', projectId] }),
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const launch = useMutation({
    mutationFn: () =>
      api.post<TrainRunOut>('/training/runs', {
        name: runName.trim() || null,
        project_id: projectId,
        dataset: datasetToken,
        base_model_id: baseModel,
        device: device === 'auto' ? null : device,
        params: {
          epochs: Number(epochs),
          imgsz: Number(imgsz),
          batch: Number(batch),
          patience: Number(patience),
          ...(lr0 !== '' ? { lr0: Number(lr0) } : {}),
          ...(optimizer ? { optimizer } : {}),
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
      navigate(`/projects/${projectId}/history/${r.id}`)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const datasetOptions = useMemo(
    () =>
      datasets.data?.map((d) => ({
        value: d.dataset,
        label: `${d.source === 'upload' ? '📦 ' : ''}${d.name} (train ${d.train} · val ${d.val} · ${d.classes}cls)${d.auto_delete ? ' · 🗑 auto' : ''}`,
      })) ?? [],
    [datasets.data],
  )

  const selectedDataset = datasets.data?.find((d) => d.dataset === datasetToken)

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
              <Group gap="xs" align="center">
                <Select
                  w={420}
                  placeholder={
                    datasetOptions.length
                      ? 'Select a dataset'
                      : 'Upload a zip or create an export in Dataset'
                  }
                  data={datasetOptions}
                  value={datasetToken}
                  onChange={setDatasetToken}
                />
                {selectedDataset?.source === 'upload' && (
                  <ActionIcon
                    color="red"
                    variant="light"
                    size="lg"
                    title="Delete this uploaded dataset"
                    loading={removeDataset.isPending}
                    onClick={() => {
                      if (
                        window.confirm(`Delete uploaded dataset "${selectedDataset.name}"? This removes it from disk.`)
                      ) {
                        removeDataset.mutate(selectedDataset.dataset.split(':')[1])
                      }
                    }}
                  >
                    <IconTrash size={18} />
                  </ActionIcon>
                )}
              </Group>
              {/* auto-delete applies only to uploaded (external) datasets */}
              {selectedDataset?.source === 'upload' && (
                <Checkbox
                  w={420}
                  checked={!!selectedDataset.auto_delete}
                  disabled={toggleAutoDelete.isPending}
                  onChange={(e) =>
                    toggleAutoDelete.mutate({
                      id: selectedDataset.dataset.split(':')[1],
                      value: e.currentTarget.checked,
                    })
                  }
                  label="Delete this dataset after a successful run"
                  description="For one-shot external zips — the original stays on your machine."
                />
              )}
              <Dropzone
                onDrop={(files) => files[0] && startDatasetUpload(files[0])}
                accept={['application/zip', 'application/x-zip-compressed']}
                multiple={false}
                loading={uploadBusy}
                radius="md"
                p="sm"
                w={420}
              >
                <Group gap="xs" justify="center" style={{ pointerEvents: 'none' }} py={4}>
                  <IconFileZip size={20} stroke={1.5} />
                  <Text size="sm">Or upload a YOLO dataset zip (with data.yaml)</Text>
                </Group>
              </Dropzone>
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
              <Select
                label="Device"
                data={['auto', 'cpu', 'mps', '0']}
                value={device}
                onChange={setDevice}
              />
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
                  />
                  <Select
                    label="Optimizer"
                    placeholder="auto"
                    data={['SGD', 'Adam', 'AdamW', 'RMSProp']}
                    value={optimizer}
                    onChange={setOptimizer}
                    clearable
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
                {datasets.data?.find((d) => d.dataset === datasetToken)?.name}
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

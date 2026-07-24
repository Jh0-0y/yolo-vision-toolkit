import { useMemo, useState } from 'react'
import {
  Anchor,
  Badge,
  Button,
  Card,
  Group,
  NumberInput,
  Select,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconFileZip, IconPlayerPlay } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import {
  api,
  uploadTrainDataset,
  type ModelOut,
  type TrainDataset,
  type TrainRunOut,
} from '../api/client'

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
  const [lr0, setLr0] = useState<number | string>('')
  const [optimizer, setOptimizer] = useState<string | null>(null)

  const datasets = useQuery({
    queryKey: ['train-datasets', projectId],
    queryFn: () => api.get<TrainDataset[]>(`/training/datasets?project_id=${projectId}`),
  })
  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })

  const uploadZip = useMutation({
    mutationFn: (file: File) => uploadTrainDataset(file),
    onSuccess: (d) => {
      notifications.show({
        message: `Dataset registered: ${d.name} (train ${d.train} · val ${d.val})`,
        color: 'green',
      })
      queryClient.invalidateQueries({ queryKey: ['train-datasets', projectId] })
      setDatasetToken(d.dataset)
    },
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
        label: `${d.source === 'upload' ? '📦 ' : ''}${d.name} (train ${d.train} · val ${d.val} · ${d.classes}cls)`,
      })) ?? [],
    [datasets.data],
  )

  return (
    <Stack gap="lg">
      <div>
        <Title order={3}>Train</Title>
        <Text c="dimmed" size="sm">
          Pick a dataset and base model — starting a run takes you to Training History.
        </Text>
      </div>

      <Card withBorder radius="md" padding="lg">
        <Stack gap="md">
          <div>
            <Text fw={600} mb={4}>
              1. Dataset
            </Text>
            <Stack gap="xs">
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
              <Dropzone
                onDrop={(files) => files[0] && uploadZip.mutate(files[0])}
                accept={['application/zip', 'application/x-zip-compressed']}
                multiple={false}
                loading={uploadZip.isPending}
                radius="md"
                p="sm"
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
              <Group grow mt="xs">
                <NumberInput label="Patience" value={patience} onChange={setPatience} min={0} />
                <NumberInput
                  label="Initial LR (lr0)"
                  value={lr0}
                  onChange={setLr0}
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

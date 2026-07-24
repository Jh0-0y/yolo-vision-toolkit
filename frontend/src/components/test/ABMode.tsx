import { useEffect, useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ActionIcon, Badge, Button, Card, Grid, Group, Select, Stack, Text, Tooltip } from '@mantine/core'
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { IconPhoto, IconX } from '@tabler/icons-react'
import {
  listResidents,
  predict,
  unloadResident,
  type ModelOut,
  type PredictResponse,
} from '../../api/client'
import type { TestConfig } from './TestControls'
import BoxOverlay from './BoxOverlay'
import DatasetPicker from './DatasetPicker'

type ImageSource =
  | { kind: 'upload'; file: File; url: string; label: string }
  | { kind: 'dataset'; name: string; url: string; label: string }

interface Props {
  projectId: string
  cfg: TestConfig
  models: ModelOut[]
}

export default function ABMode({ projectId, cfg, models }: Props) {
  const qc = useQueryClient()
  const [source, setSource] = useState<ImageSource | null>(null)
  const [modelA, setModelA] = useState<string | null>(null)
  const [modelB, setModelB] = useState<string | null>(null)
  const [resA, setResA] = useState<PredictResponse | null>(null)
  const [resB, setResB] = useState<PredictResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)

  const residentsQuery = useQuery({ queryKey: ['residents-list'], queryFn: listResidents, refetchInterval: 5000 })
  const residents = residentsQuery.data ?? []
  const nameOf = useMemo(() => Object.fromEntries(models.map((m) => [m.id, m.name])), [models])

  useEffect(() => {
    if (models.length && !modelA) setModelA(models[0].id)
    if (models.length > 1 && !modelB) setModelB(models[1].id)
  }, [models, modelA, modelB])

  useEffect(() => {
    return () => {
      if (source?.kind === 'upload') URL.revokeObjectURL(source.url)
    }
  }, [source])

  function onDrop(files: File[]) {
    const file = files[0]
    if (!file) return
    if (source?.kind === 'upload') URL.revokeObjectURL(source.url)
    setSource({ kind: 'upload', file, url: URL.createObjectURL(file), label: file.name })
    setResA(null)
    setResB(null)
  }

  async function runOne(modelId: string): Promise<PredictResponse> {
    return predict({
      modelIds: [modelId],
      projectId,
      params: { conf: cfg.conf, iou_wbf: cfg.iou, imgsz: Number(cfg.imgsz), device: cfg.device === 'auto' ? null : cfg.device },
      file: source?.kind === 'upload' ? source.file : undefined,
      imageProjectId: source?.kind === 'dataset' ? projectId : undefined,
      imageName: source?.kind === 'dataset' ? source.name : undefined,
    })
  }

  async function run() {
    if (!source || !modelA || !modelB) return
    setRunning(true)
    try {
      const [a, b] = await Promise.all([runOne(modelA), runOne(modelB)])
      setResA(a)
      setResB(b)
      qc.invalidateQueries({ queryKey: ['residents-list'] })
    } catch (e) {
      notifications.show({ color: 'red', message: `추론 실패: ${(e as Error).message}` })
    } finally {
      setRunning(false)
    }
  }

  async function unload(modelId: string) {
    await unloadResident(modelId)
    qc.invalidateQueries({ queryKey: ['residents-list'] })
  }

  const selectData = models.map((m) => ({ value: m.id, label: m.name }))

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Group align="flex-end" gap="md" wrap="wrap">
          <Select label="모델 A" data={selectData} value={modelA} onChange={setModelA} allowDeselect={false} w={200} />
          <Select label="모델 B" data={selectData} value={modelB} onChange={setModelB} allowDeselect={false} w={200} />
          {!source ? (
            <Group gap="xs" flex={1}>
              <Dropzone onDrop={onDrop} accept={IMAGE_MIME_TYPE} multiple={false} style={{ flex: 1 }}>
                <Group gap="xs" justify="center" py="sm">
                  <Dropzone.Idle><IconPhoto size={22} stroke={1.2} /></Dropzone.Idle>
                  <Dropzone.Reject><IconX size={22} /></Dropzone.Reject>
                  <Text size="sm">이미지 업로드</Text>
                </Group>
              </Dropzone>
              <Button variant="light" onClick={() => setPickerOpen(true)}>데이터셋</Button>
            </Group>
          ) : (
            <Group gap="xs">
              <Text size="sm" truncate="end" maw={220}>{source.label}</Text>
              <Button size="sm" onClick={run} loading={running} disabled={!modelA || !modelB}>비교 실행</Button>
              <Button size="sm" variant="subtle" onClick={() => { setSource(null); setResA(null); setResB(null) }}>변경</Button>
            </Group>
          )}
        </Group>

        {residents.length > 0 && (
          <Group gap="xs" mt="sm">
            <Text size="xs" c="dimmed">상주 모델:</Text>
            {residents.map((r) => (
              <Badge
                key={r.model_id}
                variant="light"
                rightSection={
                  <Tooltip label="언로드">
                    <ActionIcon size="xs" variant="transparent" color="gray" onClick={() => unload(r.model_id)}>
                      <IconX size={11} />
                    </ActionIcon>
                  </Tooltip>
                }
              >
                {nameOf[r.model_id] ?? r.model_id}
                {r.weight_mb ? ` · ${r.weight_mb}MB` : ''} · {r.device}
              </Badge>
            ))}
          </Group>
        )}
      </Card>

      {source && (
        <Grid gutter="md">
          {([['A', modelA, resA], ['B', modelB, resB]] as const).map(([tag, mid, res]) => {
            const boxes = (res?.boxes ?? []).filter((b) => b.score >= cfg.conf)
            return (
              <Grid.Col span={{ base: 12, md: 6 }} key={tag}>
                <Card withBorder radius="md" padding="sm">
                  <Group justify="space-between" mb="xs">
                    <Group gap={6}>
                      <Badge>{tag}</Badge>
                      <Text size="sm" fw={600}>{mid ? nameOf[mid] : '—'}</Text>
                    </Group>
                    {res && <Badge variant="light" color="teal">{boxes.length} boxes · {res.device}</Badge>}
                  </Group>
                  <BoxOverlay src={source.url} boxes={boxes} showLabels={cfg.showLabels} />
                </Card>
              </Grid.Col>
            )
          })}
        </Grid>
      )}

      <DatasetPicker
        projectId={projectId}
        opened={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onPick={(img) => {
          setSource({ kind: 'dataset', name: img.name, url: img.url, label: img.name })
          setResA(null)
          setResB(null)
          setPickerOpen(false)
        }}
      />
    </Stack>
  )
}

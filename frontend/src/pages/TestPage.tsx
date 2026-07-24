import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Alert,
  Badge,
  Button,
  Card,
  Checkbox,
  Grid,
  Group,
  Image,
  Loader,
  Modal,
  ScrollArea,
  SegmentedControl,
  Select,
  SimpleGrid,
  Slider,
  Stack,
  Switch,
  Text,
  Title,
} from '@mantine/core'
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { IconAlertTriangle, IconFlask, IconPhoto, IconUpload, IconX } from '@tabler/icons-react'
import {
  api,
  getResources,
  listImages,
  predict,
  type ImageItem,
  type ModelOut,
  type PredictResponse,
} from '../api/client'
import BoxOverlay, { boxKey, clsColor } from '../components/test/BoxOverlay'

type ImageSource =
  | { kind: 'upload'; file: File; url: string; label: string }
  | { kind: 'dataset'; name: string; url: string; label: string }

const IMGSZ_OPTIONS = ['320', '480', '640', '960', '1280']

export default function TestPage() {
  const { projectId = '' } = useParams()

  const modelsQuery = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })
  const resourcesQuery = useQuery({
    queryKey: ['resources'],
    queryFn: getResources,
    refetchInterval: 5000,
  })

  const [selected, setSelected] = useState<string[]>([])
  const [device, setDevice] = useState('auto')
  const [iou, setIou] = useState(0.55)
  const [imgsz, setImgsz] = useState('640')
  const [conf, setConf] = useState(0.4) // client-side filter (no re-run)

  const [source, setSource] = useState<ImageSource | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [hovered, setHovered] = useState<string | null>(null)
  const [hidden, setHidden] = useState<Set<number>>(new Set())
  const [showLabels, setShowLabels] = useState(true)
  const [pickerOpen, setPickerOpen] = useState(false)

  const models = modelsQuery.data ?? []
  const resources = resourcesQuery.data

  useEffect(() => {
    if (models.length && selected.length === 0) setSelected([models[0].id])
  }, [models, selected.length])

  useEffect(() => {
    return () => {
      if (source?.kind === 'upload') URL.revokeObjectURL(source.url)
    }
  }, [source])

  const deviceOptions = useMemo(() => {
    const opts = ['auto', 'cpu']
    const accel = resources?.accelerator
    if (accel === 'mps' || accel === 'cuda') opts.splice(1, 0, accel)
    return opts.map((v) => ({ label: v.toUpperCase(), value: v }))
  }, [resources?.accelerator])

  function onDrop(files: File[]) {
    const file = files[0]
    if (!file) return
    if (source?.kind === 'upload') URL.revokeObjectURL(source.url)
    setSource({ kind: 'upload', file, url: URL.createObjectURL(file), label: file.name })
    setResult(null)
  }

  async function run() {
    if (!selected.length) {
      notifications.show({ color: 'red', message: '모델을 하나 이상 선택하세요' })
      return
    }
    if (!source) {
      notifications.show({ color: 'red', message: '이미지를 먼저 선택하세요' })
      return
    }
    setRunning(true)
    try {
      const res = await predict({
        modelIds: selected,
        projectId,
        params: {
          conf,
          iou_wbf: iou,
          imgsz: Number(imgsz),
          device: device === 'auto' ? null : device,
        },
        file: source.kind === 'upload' ? source.file : undefined,
        imageProjectId: source.kind === 'dataset' ? projectId : undefined,
        imageName: source.kind === 'dataset' ? source.name : undefined,
      })
      setResult(res)
      setHidden(new Set())
    } catch (e) {
      notifications.show({ color: 'red', message: `추론 실패: ${(e as Error).message}` })
    } finally {
      setRunning(false)
    }
  }

  const visible = useMemo(
    () => (result?.boxes ?? []).filter((b) => b.score >= conf && !hidden.has(b.cls)),
    [result, conf, hidden],
  )
  const classCounts = useMemo(() => {
    const counts = new Map<number, number>()
    for (const b of result?.boxes ?? []) {
      if (b.score >= conf) counts.set(b.cls, (counts.get(b.cls) ?? 0) + 1)
    }
    return counts
  }, [result, conf])

  function toggleClass(cls: number) {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(cls)) next.delete(cls)
      else next.add(cls)
      return next
    })
  }

  return (
    <Stack gap="md" mt="md">
      <Group justify="space-between" align="center">
        <Group gap="xs">
          <IconFlask size={22} />
          <Title order={3}>Test</Title>
          <Text c="dimmed" size="sm">
            학습한 모델을 이미지에 바로 돌려보세요 (저장되지 않음)
          </Text>
        </Group>
        {resources && (
          <Badge variant="light" color={resources.training_active ? 'orange' : 'gray'}>
            {resources.device_label}
            {resources.resident_models > 0 ? ` · 상주 ${resources.resident_models}` : ''}
          </Badge>
        )}
      </Group>

      {resources?.warning && (
        <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="리소스 주의">
          {resources.warning}
        </Alert>
      )}

      <Grid gutter="md">
        {/* ---------- left: controls ---------- */}
        <Grid.Col span={{ base: 12, md: 3 }}>
          <Card withBorder radius="md" padding="md">
            <Stack gap="sm">
              <Text fw={600} size="sm">
                모델{' '}
                {selected.length > 1 && (
                  <Badge size="xs" ml={4}>
                    앙상블 {selected.length}
                  </Badge>
                )}
              </Text>
              {modelsQuery.isLoading ? (
                <Loader size="sm" />
              ) : models.length === 0 ? (
                <Text c="dimmed" size="sm">
                  이 프로젝트에 모델이 없습니다. 먼저 학습하거나 업로드하세요.
                </Text>
              ) : (
                <Checkbox.Group value={selected} onChange={setSelected}>
                  <Stack gap={6}>
                    {models.map((m) => (
                      <Checkbox
                        key={m.id}
                        value={m.id}
                        label={
                          <Group gap={6} wrap="nowrap">
                            <Text size="sm">{m.name}</Text>
                            <Badge size="xs" variant="light" color="gray">
                              {m.task}
                            </Badge>
                          </Group>
                        }
                      />
                    ))}
                  </Stack>
                </Checkbox.Group>
              )}

              <Text fw={600} size="sm" mt="xs">
                Device
              </Text>
              <SegmentedControl
                size="xs"
                data={deviceOptions}
                value={device}
                onChange={setDevice}
                fullWidth
              />

              <Text fw={600} size="sm" mt="xs">
                Confidence <Text span c="dimmed">{conf.toFixed(2)}</Text>
              </Text>
              <Slider min={0} max={1} step={0.01} value={conf} onChange={setConf} label={(v) => v.toFixed(2)} />

              <Text fw={600} size="sm" mt="xs">
                IoU (WBF) <Text span c="dimmed">{iou.toFixed(2)}</Text>
              </Text>
              <Slider min={0.1} max={0.95} step={0.05} value={iou} onChange={setIou} label={(v) => v.toFixed(2)} />

              <Select
                mt="xs"
                label="Image size"
                size="xs"
                data={IMGSZ_OPTIONS}
                value={imgsz}
                onChange={(v) => v && setImgsz(v)}
                allowDeselect={false}
              />

              <Button mt="sm" onClick={run} loading={running} disabled={!source || !selected.length}>
                추론 실행
              </Button>
              <Text c="dimmed" size="xs">
                Confidence는 재추론 없이 즉시 필터링됩니다. Device·IoU·Image size 변경 후엔 다시 실행하세요.
              </Text>
            </Stack>
          </Card>
        </Grid.Col>

        {/* ---------- center: image + overlay ---------- */}
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Card withBorder radius="md" padding="md">
            {!source ? (
              <Stack gap="sm">
                <Dropzone onDrop={onDrop} accept={IMAGE_MIME_TYPE} multiple={false}>
                  <Stack align="center" gap="xs" py="xl">
                    <Dropzone.Accept>
                      <IconUpload size={40} />
                    </Dropzone.Accept>
                    <Dropzone.Reject>
                      <IconX size={40} />
                    </Dropzone.Reject>
                    <Dropzone.Idle>
                      <IconPhoto size={40} stroke={1.2} />
                    </Dropzone.Idle>
                    <Text size="sm">이미지를 드래그하거나 클릭해서 업로드</Text>
                  </Stack>
                </Dropzone>
                <Button variant="light" onClick={() => setPickerOpen(true)}>
                  데이터셋에서 선택
                </Button>
              </Stack>
            ) : (
              <Stack gap="xs">
                <Group justify="space-between">
                  <Text size="sm" truncate="end" maw={320}>
                    {source.label}
                  </Text>
                  <Group gap="xs">
                    <Switch
                      size="xs"
                      label="라벨"
                      checked={showLabels}
                      onChange={(e) => setShowLabels(e.currentTarget.checked)}
                    />
                    <Button
                      size="xs"
                      variant="subtle"
                      onClick={() => {
                        setSource(null)
                        setResult(null)
                      }}
                    >
                      다른 이미지
                    </Button>
                  </Group>
                </Group>
                <BoxOverlay
                  src={source.url}
                  boxes={visible}
                  hovered={hovered}
                  onHover={setHovered}
                  showLabels={showLabels}
                />
                {result && (
                  <Group gap="xs" mt={4}>
                    <Badge variant="light">device {result.device}</Badge>
                    <Badge variant="light" color="teal">
                      {visible.length}/{result.boxes.length} boxes
                    </Badge>
                    <Text c="dimmed" size="xs">
                      floor {result.floor}
                    </Text>
                  </Group>
                )}
              </Stack>
            )}
          </Card>
        </Grid.Col>

        {/* ---------- right: detections ---------- */}
        <Grid.Col span={{ base: 12, md: 3 }}>
          <Card withBorder radius="md" padding="md">
            <Text fw={600} size="sm" mb="xs">
              검출 결과
            </Text>
            {!result ? (
              <Text c="dimmed" size="sm">
                추론을 실행하면 결과가 여기에 표시됩니다.
              </Text>
            ) : result.boxes.length === 0 ? (
              <Text c="dimmed" size="sm">
                검출된 객체가 없습니다.
              </Text>
            ) : (
              <Stack gap="xs">
                <Text c="dimmed" size="xs">
                  클래스 (클릭해서 표시 토글)
                </Text>
                {[...classCounts.entries()].map(([cls, count]) => (
                  <Group
                    key={cls}
                    gap="xs"
                    justify="space-between"
                    onClick={() => toggleClass(cls)}
                    style={{ cursor: 'pointer', opacity: hidden.has(cls) ? 0.4 : 1 }}
                  >
                    <Group gap={6}>
                      <span style={{ width: 12, height: 12, borderRadius: 3, background: clsColor(cls) }} />
                      <Text size="sm">{result.names[cls] ?? cls}</Text>
                    </Group>
                    <Badge size="sm" variant="light">
                      {count}
                    </Badge>
                  </Group>
                ))}

                <Text c="dimmed" size="xs" mt="xs">
                  박스 ({visible.length})
                </Text>
                <ScrollArea.Autosize mah={320}>
                  <Stack gap={4}>
                    {visible.map((b) => {
                      const key = boxKey(b)
                      return (
                        <Group
                          key={key}
                          gap="xs"
                          justify="space-between"
                          onMouseEnter={() => setHovered(key)}
                          onMouseLeave={() => setHovered(null)}
                          style={{
                            cursor: 'default',
                            background: hovered === key ? 'var(--mantine-color-default-hover)' : undefined,
                            borderRadius: 4,
                            padding: '2px 6px',
                          }}
                        >
                          <Group gap={6}>
                            <span style={{ width: 10, height: 10, borderRadius: 2, background: clsColor(b.cls) }} />
                            <Text size="sm">{b.name}</Text>
                            {b.agree > 1 && (
                              <Badge size="xs" color="grape" variant="light">
                                ×{b.agree}
                              </Badge>
                            )}
                          </Group>
                          <Text size="sm" c="dimmed">
                            {(b.score * 100).toFixed(0)}%
                          </Text>
                        </Group>
                      )
                    })}
                  </Stack>
                </ScrollArea.Autosize>
              </Stack>
            )}
          </Card>
        </Grid.Col>
      </Grid>

      <DatasetPicker
        projectId={projectId}
        opened={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onPick={(img) => {
          setSource({ kind: 'dataset', name: img.name, url: img.url, label: img.name })
          setResult(null)
          setPickerOpen(false)
        }}
      />
    </Stack>
  )
}

function DatasetPicker({
  projectId,
  opened,
  onClose,
  onPick,
}: {
  projectId: string
  opened: boolean
  onClose: () => void
  onPick: (img: ImageItem) => void
}) {
  const query = useQuery({
    queryKey: ['test-picker', projectId],
    queryFn: () => listImages(projectId, { size: 60, sort: 'created', order: 'desc' }),
    enabled: opened,
  })
  return (
    <Modal opened={opened} onClose={onClose} title="데이터셋에서 이미지 선택" size="xl">
      {query.isLoading ? (
        <Loader />
      ) : (
        <SimpleGrid cols={{ base: 3, sm: 5 }} spacing="xs">
          {(query.data?.items ?? []).map((img) => (
            <Image
              key={img.name}
              src={img.thumb}
              radius="sm"
              style={{ cursor: 'pointer', aspectRatio: '1', objectFit: 'cover' }}
              onClick={() => onPick(img)}
            />
          ))}
        </SimpleGrid>
      )}
    </Modal>
  )
}

import { useEffect, useMemo, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  Grid,
  Group,
  ScrollArea,
  Stack,
  Text,
} from '@mantine/core'
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { IconPhoto, IconUpload, IconX } from '@tabler/icons-react'
import { predict, type PredictResponse } from '../../api/client'
import type { TestConfig } from './TestControls'
import BoxOverlay, { boxKey, clsColor } from './BoxOverlay'
import DatasetPicker from './DatasetPicker'

type ImageSource =
  | { kind: 'upload'; file: File; url: string; label: string }
  | { kind: 'dataset'; name: string; url: string; label: string }

export default function SingleMode({ projectId, cfg }: { projectId: string; cfg: TestConfig }) {
  const [source, setSource] = useState<ImageSource | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [hovered, setHovered] = useState<string | null>(null)
  const [hidden, setHidden] = useState<Set<number>>(new Set())
  const [pickerOpen, setPickerOpen] = useState(false)

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
    setResult(null)
  }

  async function run() {
    if (!cfg.selected.length) {
      notifications.show({ color: 'red', message: '모델을 하나 이상 선택하세요' })
      return
    }
    if (!source) return
    setRunning(true)
    try {
      const res = await predict({
        modelIds: cfg.selected,
        projectId,
        params: { conf: cfg.conf, iou_wbf: cfg.iou, imgsz: Number(cfg.imgsz), device: cfg.device === 'auto' ? null : cfg.device },
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
    () => (result?.boxes ?? []).filter((b) => b.score >= cfg.conf && !hidden.has(b.cls)),
    [result, cfg.conf, hidden],
  )
  const classCounts = useMemo(() => {
    const counts = new Map<number, number>()
    for (const b of result?.boxes ?? []) {
      if (b.score >= cfg.conf) counts.set(b.cls, (counts.get(b.cls) ?? 0) + 1)
    }
    return counts
  }, [result, cfg.conf])

  function toggleClass(cls: number) {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(cls)) next.delete(cls)
      else next.add(cls)
      return next
    })
  }

  return (
    <Grid gutter="md">
      <Grid.Col span={{ base: 12, md: 8 }}>
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
                  <Button size="xs" onClick={run} loading={running} disabled={!cfg.selected.length}>
                    추론 실행
                  </Button>
                  <Button size="xs" variant="subtle" onClick={() => { setSource(null); setResult(null) }}>
                    다른 이미지
                  </Button>
                </Group>
              </Group>
              <BoxOverlay src={source.url} boxes={visible} hovered={hovered} onHover={setHovered} showLabels={cfg.showLabels} />
              {result && (
                <Group gap="xs" mt={4}>
                  <Badge variant="light">device {result.device}</Badge>
                  <Badge variant="light" color="teal">
                    {visible.length}/{result.boxes.length} boxes
                  </Badge>
                  <Text c="dimmed" size="xs">floor {result.floor}</Text>
                </Group>
              )}
            </Stack>
          )}
        </Card>
      </Grid.Col>

      <Grid.Col span={{ base: 12, md: 4 }}>
        <Card withBorder radius="md" padding="md">
          <Text fw={600} size="sm" mb="xs">
            검출 결과
          </Text>
          {!result ? (
            <Text c="dimmed" size="sm">추론을 실행하면 결과가 여기에 표시됩니다.</Text>
          ) : result.boxes.length === 0 ? (
            <Text c="dimmed" size="sm">검출된 객체가 없습니다.</Text>
          ) : (
            <Stack gap="xs">
              <Text c="dimmed" size="xs">클래스 (클릭해서 표시 토글)</Text>
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
                  <Badge size="sm" variant="light">{count}</Badge>
                </Group>
              ))}
              <Text c="dimmed" size="xs" mt="xs">박스 ({visible.length})</Text>
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
                          background: hovered === key ? 'var(--mantine-color-default-hover)' : undefined,
                          borderRadius: 4,
                          padding: '2px 6px',
                        }}
                      >
                        <Group gap={6}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: clsColor(b.cls) }} />
                          <Text size="sm">{b.name}</Text>
                          {b.agree > 1 && <Badge size="xs" color="grape" variant="light">×{b.agree}</Badge>}
                        </Group>
                        <Text size="sm" c="dimmed">{(b.score * 100).toFixed(0)}%</Text>
                      </Group>
                    )
                  })}
                </Stack>
              </ScrollArea.Autosize>
            </Stack>
          )}
        </Card>
      </Grid.Col>

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
    </Grid>
  )
}

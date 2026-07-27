import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Modal,
  MultiSelect,
  NumberInput,
  Progress,
  ScrollArea,
  SimpleGrid,
  Slider,
  Stack,
  Text,
} from '@mantine/core'
import { BarChart } from '@mantine/charts'
import { IconAlertTriangle } from '@tabler/icons-react'
import {
  getCompareResult,
  listImages,
  startCompare,
  subscribeCompareEvents,
  type CompareBox,
  type CompareImage,
  type CompareProgress,
  type CompareResult,
  type ImageItem,
  type ModelOut,
} from '../../api/client'
import ImageGrid from '../dataset/ImageGrid'
import StatTile from '../StatTile'

const MIN = 9
const MAX = 27
const GT_COLOR = '#51cf66' // ground truth = green (dashed)
const MODEL_COLORS = ['#4dabf7', '#f783ac', '#ffa94d', '#845ef7', '#38d9a9', '#ff8787']

interface Props {
  projectId: string
  models: ModelOut[]
}

interface Layer {
  boxes: CompareBox[]
  color: string
  dashed: boolean
}

function BoxOverlay({ src, layers }: { src: string; layers: Layer[] }) {
  return (
    <div style={{ position: 'relative', width: '100%', lineHeight: 0 }}>
      <img src={src} alt="" style={{ width: '100%', display: 'block', borderRadius: 6 }} />
      {layers.flatMap((layer, li) =>
        layer.boxes.map((b, i) => {
          const [x1, y1, x2, y2] = b.xyxyn
          return (
            <div
              key={`${li}-${i}`}
              style={{
                position: 'absolute',
                left: `${x1 * 100}%`,
                top: `${y1 * 100}%`,
                width: `${(x2 - x1) * 100}%`,
                height: `${(y2 - y1) * 100}%`,
                border: `2px ${layer.dashed ? 'dashed' : 'solid'} ${layer.color}`,
                borderRadius: 2,
              }}
            />
          )
        }),
      )}
    </div>
  )
}

/** Score 1..N models against ground truth on a chosen set of labeled images
 *  (9–27), then compare per-model metrics and overlay each model's boxes. */
export default function CompareMode({ projectId, models }: Props) {
  const [modelIds, setModelIds] = useState<string[]>([])
  const [conf, setConf] = useState(0.4)
  const [iou, setIou] = useState(0.5)
  const [imgsz, setImgsz] = useState<number | string>(640)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<CompareProgress | null>(null)
  const [result, setResult] = useState<CompareResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [enlarged, setEnlarged] = useState<CompareImage | null>(null)
  const unsub = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (models.length && !modelIds.length) setModelIds([models[0].id])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])
  useEffect(() => () => unsub.current?.(), [])

  const imagesQuery = useQuery({
    queryKey: ['compare-images', projectId],
    queryFn: () => listImages(projectId, { labeled: true, size: 60, sort: 'created', order: 'desc' }),
  })
  const items: ImageItem[] = imagesQuery.data?.items ?? []

  const colorOf = (modelId: string) =>
    MODEL_COLORS[modelIds.indexOf(modelId) % MODEL_COLORS.length]

  const onSelect = (next: Set<string>) => {
    if (next.size <= MAX) setSelected(next)
  }

  async function run() {
    unsub.current?.()
    setError(null)
    setResult(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      const { job_id } = await startCompare({
        projectId,
        modelIds,
        imageNames: [...selected],
        params: { conf, iou_wbf: iou, imgsz: Number(imgsz), device: null },
      })
      unsub.current = subscribeCompareEvents(job_id, async (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          try {
            setResult(await getCompareResult(job_id))
          } catch (e) {
            setError((e as Error).message)
          }
          setRunning(false)
        } else if (ev.phase === 'error') {
          setError(ev.msg || 'Comparison failed')
          setRunning(false)
        }
      })
    } catch (e) {
      setError((e as Error).message)
      setRunning(false)
    }
  }

  const pct =
    progress?.total && progress.done != null ? Math.round((progress.done / progress.total) * 100) : 0

  const bestId = useMemo(() => {
    if (!result?.per_model.length) return null
    return result.per_model.reduce((a, b) => (b.overall.f1 > a.overall.f1 ? b : a)).model_id
  }, [result])

  const barData = useMemo(() => {
    if (!result) return []
    return (['precision', 'recall', 'f1'] as const).map((k) => ({
      metric: k[0].toUpperCase() + k.slice(1),
      ...Object.fromEntries(result.per_model.map((m) => [m.name, m.overall[k]])),
    }))
  }, [result])

  const canRun = modelIds.length > 0 && selected.size >= MIN && !running

  return (
    <Stack gap="md">
      {/* settings */}
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <Text size="sm" fw={600}>
            Settings
          </Text>
          <MultiSelect
            label="Models to compare (1 or more)"
            placeholder={models.length ? 'Pick models' : 'No models — train or upload one first'}
            data={models.map((m) => ({ value: m.id, label: m.name }))}
            value={modelIds}
            onChange={setModelIds}
            disabled={running || !models.length}
          />
          <Group grow align="flex-start">
            <div>
              <Text size="sm" fw={600}>
                Confidence <Text span c="dimmed">{conf.toFixed(2)}</Text>
              </Text>
              <Slider min={0.05} max={0.95} step={0.05} value={conf} onChange={setConf} disabled={running} />
            </div>
            <div>
              <Text size="sm" fw={600}>
                Match IoU <Text span c="dimmed">{iou.toFixed(2)}</Text>
              </Text>
              <Slider min={0.3} max={0.9} step={0.05} value={iou} onChange={setIou} disabled={running} />
            </div>
            <NumberInput
              label="Image size"
              value={imgsz}
              onChange={setImgsz}
              min={64}
              step={32}
              disabled={running}
            />
          </Group>
        </Stack>
      </Card>

      {/* image picker */}
      <Card withBorder radius="md" padding="md">
        <Group justify="space-between" mb="xs">
          <div>
            <Text size="sm" fw={600}>
              Pick labeled images ({MIN}–{MAX})
            </Text>
            <Text size="xs" c={selected.size >= MIN ? 'dimmed' : 'orange'}>
              {selected.size} selected{selected.size < MIN ? ` — pick at least ${MIN}` : ''}
              {selected.size >= MAX ? ' — maximum reached' : ''}
            </Text>
          </div>
          <Button onClick={run} disabled={!canRun} loading={running}>
            Run comparison
          </Button>
        </Group>

        {items.length === 0 ? (
          <Text c="dimmed" py="lg" ta="center">
            No labeled images in this project yet. Label some images first.
          </Text>
        ) : (
          <ScrollArea.Autosize mah={360}>
            <ImageGrid items={items} selected={selected} onSelectedChange={onSelect} onOpen={() => {}} />
          </ScrollArea.Autosize>
        )}
      </Card>

      {error && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />} withCloseButton onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {running && (
        <Stack gap={4}>
          <Text size="sm">
            Scoring… {progress?.done ?? 0}/{progress?.total ?? '?'} images
          </Text>
          <Progress value={pct} animated />
        </Stack>
      )}

      {result && !running && (
        <Stack gap="md">
          {result.warning && (
            <Alert color="orange" icon={<IconAlertTriangle size={18} />}>
              {result.warning}
            </Alert>
          )}

          {/* per-model metrics */}
          <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
            {result.per_model.map((m) => (
              <Card key={m.model_id} withBorder radius="md" padding="sm">
                <Group justify="space-between" mb={4}>
                  <Group gap={6}>
                    <span
                      style={{ width: 10, height: 10, borderRadius: 2, background: colorOf(m.model_id) }}
                    />
                    <Text size="sm" fw={600} truncate="end" maw={140}>
                      {m.name}
                    </Text>
                  </Group>
                  {m.model_id === bestId && (
                    <Badge size="xs" color="teal" variant="light">
                      Best F1
                    </Badge>
                  )}
                </Group>
                <Group grow>
                  <StatTile label="P" value={m.overall.precision.toFixed(3)} color="indigo.6" />
                  <StatTile label="R" value={m.overall.recall.toFixed(3)} color="orange.6" />
                  <StatTile label="F1" value={m.overall.f1.toFixed(3)} color="teal.6" />
                </Group>
                <Text size="xs" c="dimmed" mt={6}>
                  {m.detections} detections · TP {m.overall.tp} · FP {m.overall.fp} · FN {m.overall.fn}
                </Text>
              </Card>
            ))}
          </SimpleGrid>

          <Card withBorder radius="md" padding="md">
            <Text size="sm" fw={600} mb="xs">
              Precision / Recall / F1 by model
            </Text>
            <BarChart
              h={240}
              data={barData}
              dataKey="metric"
              series={result.per_model.map((m) => ({ name: m.name, color: colorOf(m.model_id) }))}
              yAxisProps={{ domain: [0, 1] }}
              withLegend
            />
          </Card>

          {/* visual comparison */}
          <Stack gap="xs">
            <Group gap="md">
              <Text size="sm" fw={600}>
                Per-image boxes
              </Text>
              <Group gap={6}>
                <span style={{ width: 14, height: 0, borderTop: `2px dashed ${GT_COLOR}` }} />
                <Text size="xs" c="dimmed">
                  Ground truth
                </Text>
                {result.per_model.map((m) => (
                  <Group gap={4} key={m.model_id}>
                    <span style={{ width: 14, height: 2, background: colorOf(m.model_id) }} />
                    <Text size="xs" c="dimmed">
                      {m.name}
                    </Text>
                  </Group>
                ))}
              </Group>
            </Group>
            <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
              {result.images.map((img) => (
                <Stack key={img.stem} gap={4} style={{ cursor: 'zoom-in' }} onClick={() => setEnlarged(img)}>
                  <BoxOverlay
                    src={img.url}
                    layers={[
                      { boxes: img.gt_boxes, color: GT_COLOR, dashed: true },
                      ...img.per_model.map((pm) => ({
                        boxes: pm.pred_boxes,
                        color: colorOf(pm.model_id),
                        dashed: false,
                      })),
                    ]}
                  />
                  <Text size="xs" c="dimmed" truncate="end">
                    {img.name}
                  </Text>
                </Stack>
              ))}
            </SimpleGrid>
          </Stack>
        </Stack>
      )}

      <Modal opened={!!enlarged} onClose={() => setEnlarged(null)} size="xl" title={enlarged?.name}>
        {enlarged && (
          <BoxOverlay
            src={enlarged.url}
            layers={[
              { boxes: enlarged.gt_boxes, color: GT_COLOR, dashed: true },
              ...enlarged.per_model.map((pm) => ({
                boxes: pm.pred_boxes,
                color: colorOf(pm.model_id),
                dashed: false,
              })),
            ]}
          />
        )}
      </Modal>
    </Stack>
  )
}

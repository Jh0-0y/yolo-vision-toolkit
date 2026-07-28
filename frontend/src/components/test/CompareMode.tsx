import { useEffect, useMemo, useRef, useState } from 'react'
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
  SimpleGrid,
  Slider,
  Stack,
  Table,
  Text,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { BarChart } from '@mantine/charts'
import { IconAlertTriangle, IconFileZip, IconX } from '@tabler/icons-react'
import {
  getCompareResult,
  startCompare,
  subscribeCompareEvents,
  type CompareBox,
  type CompareImage,
  type CompareProgress,
  type CompareResult,
  type ModelOut,
} from '../../api/client'

const ZIP_MIME = ['application/zip', 'application/x-zip-compressed', 'application/octet-stream']
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

const fmt = (v: number | undefined) => (v == null ? '—' : v.toFixed(3))

/** Score 1..N models against an uploaded YOLO test set (images + labels +
 *  data.yaml): per-model mAP + P/R/F1, per-class metrics, and box overlays. */
export default function CompareMode({ projectId, models }: Props) {
  const [modelIds, setModelIds] = useState<string[]>([])
  const [conf, setConf] = useState(0.4)
  const [iou, setIou] = useState(0.5)
  const [imgsz, setImgsz] = useState<number | string>(640)
  const [file, setFile] = useState<File | null>(null)
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

  const colorOf = (modelId: string) =>
    MODEL_COLORS[modelIds.indexOf(modelId) % MODEL_COLORS.length]

  async function run() {
    if (!file || !modelIds.length) return
    unsub.current?.()
    setError(null)
    setResult(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      const { job_id } = await startCompare({
        projectId,
        modelIds,
        file,
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
        } else if (ev.phase === 'cancelled') {
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

  // "Best" = highest mAP@0.5:0.95 (the headline COCO metric)
  const bestId = useMemo(() => {
    if (!result?.per_model.length) return null
    return result.per_model.reduce((a, b) => (b.map > a.map ? b : a)).model_id
  }, [result])

  const prfData = useMemo(() => {
    if (!result) return []
    return (['precision', 'recall', 'f1'] as const).map((k) => ({
      metric: k[0].toUpperCase() + k.slice(1),
      ...Object.fromEntries(result.per_model.map((m) => [m.name, m.overall[k]])),
    }))
  }, [result])

  const mapData = useMemo(() => {
    if (!result) return []
    return (['map50', 'map'] as const).map((k) => ({
      metric: k === 'map50' ? 'mAP@0.5' : 'mAP@0.5:0.95',
      ...Object.fromEntries(result.per_model.map((m) => [m.name, m[k]])),
    }))
  }, [result])

  const canRun = modelIds.length > 0 && !!file && !running

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

      {/* test-set upload */}
      <Card withBorder radius="md" padding="md">
        <Group justify="space-between" mb="xs">
          <div>
            <Text size="sm" fw={600}>
              Test set (YOLO dataset zip)
            </Text>
            <Text size="xs" c="dimmed">
              A .zip with <code>images/</code>, <code>labels/</code> and <code>data.yaml</code> — same as an
              Exports download. Ground-truth labels are required to score the models.
            </Text>
          </div>
          <Button onClick={run} disabled={!canRun} loading={running}>
            Run comparison
          </Button>
        </Group>

        {!file ? (
          <Dropzone
            onDrop={(files) => files[0] && setFile(files[0])}
            accept={ZIP_MIME}
            multiple={false}
            disabled={running || !modelIds.length}
          >
            <Stack align="center" gap="xs" py="xl">
              <Dropzone.Idle>
                <IconFileZip size={40} stroke={1.2} />
              </Dropzone.Idle>
              <Dropzone.Reject>
                <IconX size={40} />
              </Dropzone.Reject>
              <Text size="sm">Drop a YOLO dataset .zip or click to upload</Text>
              <Text size="xs" c="dimmed">
                Each model is scored separately so you can compare them.
              </Text>
            </Stack>
          </Dropzone>
        ) : (
          <Group justify="space-between">
            <Group gap={8}>
              <IconFileZip size={20} />
              <Text size="sm" truncate="end" maw={360}>
                {file.name}
              </Text>
            </Group>
            <Button size="xs" variant="subtle" onClick={() => setFile(null)} disabled={running}>
              Choose another
            </Button>
          </Group>
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
                <Group justify="space-between" mb={6}>
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
                      Best mAP
                    </Badge>
                  )}
                </Group>
                <Group grow mb={6}>
                  <Metric label="mAP@.5" value={m.map50.toFixed(3)} strong />
                  <Metric label="mAP@.5:.95" value={m.map.toFixed(3)} strong />
                </Group>
                <Group grow>
                  <Metric label="P" value={m.overall.precision.toFixed(3)} />
                  <Metric label="R" value={m.overall.recall.toFixed(3)} />
                  <Metric label="F1" value={m.overall.f1.toFixed(3)} />
                </Group>
                <Text size="xs" c="dimmed" mt={6}>
                  {m.detections} detections · TP {m.overall.tp} · FP {m.overall.fp} · FN {m.overall.fn}
                </Text>
              </Card>
            ))}
          </SimpleGrid>

          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
            <Card withBorder radius="md" padding="md">
              <Text size="sm" fw={600} mb="xs">
                mAP by model
              </Text>
              <BarChart
                h={240}
                data={mapData}
                dataKey="metric"
                series={result.per_model.map((m) => ({ name: m.name, color: colorOf(m.model_id) }))}
                yAxisProps={{ domain: [0, 1] }}
                withLegend
              />
            </Card>
            <Card withBorder radius="md" padding="md">
              <Text size="sm" fw={600} mb="xs">
                Precision / Recall / F1 by model
              </Text>
              <BarChart
                h={240}
                data={prfData}
                dataKey="metric"
                series={result.per_model.map((m) => ({ name: m.name, color: colorOf(m.model_id) }))}
                yAxisProps={{ domain: [0, 1] }}
                withLegend
              />
            </Card>
          </SimpleGrid>

          {/* per-class metrics, one table per model */}
          <Stack gap="xs">
            <Text size="sm" fw={600}>
              Per-class metrics
            </Text>
            <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="sm">
              {result.per_model.map((m) => (
                <Card key={m.model_id} withBorder radius="md" padding="sm">
                  <Group gap={6} mb={6}>
                    <span
                      style={{ width: 10, height: 10, borderRadius: 2, background: colorOf(m.model_id) }}
                    />
                    <Text size="sm" fw={600}>
                      {m.name}
                    </Text>
                  </Group>
                  <Table.ScrollContainer minWidth={420}>
                    <Table striped highlightOnHover fz="xs" verticalSpacing={4}>
                      <Table.Thead>
                        <Table.Tr>
                          <Table.Th>Class</Table.Th>
                          <Table.Th ta="right">GT</Table.Th>
                          <Table.Th ta="right">P</Table.Th>
                          <Table.Th ta="right">R</Table.Th>
                          <Table.Th ta="right">F1</Table.Th>
                          <Table.Th ta="right">AP@.5</Table.Th>
                          <Table.Th ta="right">AP@.5:.95</Table.Th>
                        </Table.Tr>
                      </Table.Thead>
                      <Table.Tbody>
                        {m.per_class.map((c) => (
                          <Table.Tr key={c.cls}>
                            <Table.Td>{c.name}</Table.Td>
                            <Table.Td ta="right">{c.gt}</Table.Td>
                            <Table.Td ta="right">{fmt(c.precision)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.recall)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.f1)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.ap50)}</Table.Td>
                            <Table.Td ta="right">{fmt(c.ap)}</Table.Td>
                          </Table.Tr>
                        ))}
                        {m.per_class.length === 0 && (
                          <Table.Tr>
                            <Table.Td colSpan={7}>
                              <Text size="xs" c="dimmed" ta="center">
                                No overlapping classes between this model and the test set.
                              </Text>
                            </Table.Td>
                          </Table.Tr>
                        )}
                      </Table.Tbody>
                    </Table>
                  </Table.ScrollContainer>
                </Card>
              ))}
            </SimpleGrid>
          </Stack>

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

function Metric({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <Stack gap={0} align="center">
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text size={strong ? 'md' : 'sm'} fw={strong ? 700 : 600}>
        {value}
      </Text>
    </Stack>
  )
}

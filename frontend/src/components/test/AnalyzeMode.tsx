import { useEffect, useRef, useState } from 'react'
import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Modal,
  Progress,
  SimpleGrid,
  Stack,
  Switch,
  Table,
  Text,
} from '@mantine/core'
import { IconAlertTriangle } from '@tabler/icons-react'
import {
  getAnalyzeResult,
  startAnalyze,
  subscribeAnalyzeEvents,
  type AnalyzeBox,
  type AnalyzeProgress,
  type AnalyzeResult,
  type WorstImage,
} from '../../api/client'
import type { TestConfig } from './TestControls'
import { clsColor } from './BoxOverlay'

const GT_COLOR = '#51cf66' // ground truth = green (dashed)

function DiffOverlay({ src, gt, pred }: { src: string; gt: AnalyzeBox[]; pred: AnalyzeBox[] }) {
  const box = (b: AnalyzeBox, color: string, dashed: boolean, key: string) => {
    const [x1, y1, x2, y2] = b.xyxyn
    return (
      <div
        key={key}
        style={{
          position: 'absolute',
          left: `${x1 * 100}%`,
          top: `${y1 * 100}%`,
          width: `${(x2 - x1) * 100}%`,
          height: `${(y2 - y1) * 100}%`,
          border: `2px ${dashed ? 'dashed' : 'solid'} ${color}`,
          borderRadius: 2,
        }}
      />
    )
  }
  return (
    <div style={{ position: 'relative', width: '100%', lineHeight: 0 }}>
      <img src={src} alt="" style={{ width: '100%', display: 'block', borderRadius: 6 }} />
      {gt.map((b, i) => box(b, GT_COLOR, true, `g${i}`))}
      {pred.map((b, i) => box(b, clsColor(b.cls), false, `p${i}`))}
    </div>
  )
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <Card withBorder radius="md" padding="sm">
      <Text size="xs" c="dimmed">{label}</Text>
      <Text fw={700} size="xl">{value}</Text>
    </Card>
  )
}

export default function AnalyzeMode({ projectId, cfg }: { projectId: string; cfg: TestConfig }) {
  const [reviewedOnly, setReviewedOnly] = useState(false)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<AnalyzeProgress | null>(null)
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [enlarged, setEnlarged] = useState<WorstImage | null>(null)
  const unsub = useRef<(() => void) | null>(null)

  useEffect(() => () => unsub.current?.(), [])

  async function run() {
    if (!cfg.selected.length) {
      setError('모델을 하나 이상 선택하세요')
      return
    }
    unsub.current?.()
    setError(null)
    setResult(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      const { job_id } = await startAnalyze({
        projectId,
        modelIds: cfg.selected,
        params: { conf: cfg.conf, iou_wbf: cfg.iou, imgsz: Number(cfg.imgsz), device: cfg.device === 'auto' ? null : cfg.device },
        reviewedOnly,
      })
      unsub.current = subscribeAnalyzeEvents(job_id, async (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          try {
            setResult(await getAnalyzeResult(job_id))
          } catch (e) {
            setError((e as Error).message)
          }
          setRunning(false)
        } else if (ev.phase === 'error') {
          setError(ev.msg || '분석 실패')
          setRunning(false)
        }
      })
    } catch (e) {
      setError((e as Error).message)
      setRunning(false)
    }
  }

  const pct = progress?.total && progress.done != null ? Math.round((progress.done / progress.total) * 100) : 0

  return (
    <Card withBorder radius="md" padding="md">
      <Stack gap="md">
        <Group justify="space-between">
          <Group gap="md">
            <Button onClick={run} loading={running} disabled={!cfg.selected.length}>분석 실행</Button>
            <Switch
              label="검수된 이미지만"
              checked={reviewedOnly}
              onChange={(e) => setReviewedOnly(e.currentTarget.checked)}
              disabled={running}
            />
          </Group>
          <Text c="dimmed" size="sm">라벨된 데이터에 모델을 돌려 정답과 비교합니다 (conf {cfg.conf.toFixed(2)}, IoU {cfg.iou.toFixed(2)})</Text>
        </Group>

        {error && (
          <Alert color="red" icon={<IconAlertTriangle size={18} />} withCloseButton onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {running && (
          <Stack gap={4}>
            <Text size="sm">분석 중… {progress?.done ?? 0}/{progress?.total ?? '?'} 이미지</Text>
            <Progress value={pct} animated />
          </Stack>
        )}

        {result && !running && (
          <Stack gap="md">
            <SimpleGrid cols={{ base: 2, sm: 4 }}>
              <Tile label="이미지" value={String(result.images)} />
              <Tile label="Precision" value={result.overall.precision.toFixed(3)} />
              <Tile label="Recall" value={result.overall.recall.toFixed(3)} />
              <Tile label="F1" value={result.overall.f1.toFixed(3)} />
            </SimpleGrid>

            <div style={{ overflowX: 'auto' }}>
              <Table striped highlightOnHover withTableBorder>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>클래스</Table.Th>
                    <Table.Th>Precision</Table.Th>
                    <Table.Th>Recall</Table.Th>
                    <Table.Th>F1</Table.Th>
                    <Table.Th>TP</Table.Th>
                    <Table.Th>FP</Table.Th>
                    <Table.Th>FN</Table.Th>
                    <Table.Th>GT</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {result.per_class.map((r) => (
                    <Table.Tr key={r.cls}>
                      <Table.Td>
                        <Group gap={6}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: clsColor(r.cls) }} />
                          {r.name}
                        </Group>
                      </Table.Td>
                      <Table.Td>{r.precision.toFixed(3)}</Table.Td>
                      <Table.Td>{r.recall.toFixed(3)}</Table.Td>
                      <Table.Td>{r.f1.toFixed(3)}</Table.Td>
                      <Table.Td>{r.tp}</Table.Td>
                      <Table.Td>{r.fp}</Table.Td>
                      <Table.Td>{r.fn}</Table.Td>
                      <Table.Td>{r.gt}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </div>

            {result.worst.length > 0 && (
              <Stack gap="xs">
                <Group gap="md">
                  <Text fw={600} size="sm">가장 많이 틀린 이미지</Text>
                  <Group gap={6}>
                    <span style={{ width: 12, height: 2, background: GT_COLOR, display: 'inline-block' }} />
                    <Text size="xs" c="dimmed">정답(GT)</Text>
                    <span style={{ width: 12, height: 2, background: '#4dabf7', display: 'inline-block' }} />
                    <Text size="xs" c="dimmed">예측</Text>
                  </Group>
                </Group>
                <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
                  {result.worst.map((w) => (
                    <Stack key={w.stem} gap={4} style={{ cursor: 'zoom-in' }} onClick={() => setEnlarged(w)}>
                      <DiffOverlay src={w.url} gt={w.gt_boxes} pred={w.pred_boxes} />
                      <Group gap="xs">
                        {w.fn > 0 && <Badge size="xs" color="orange" variant="light">놓침 {w.fn}</Badge>}
                        {w.fp > 0 && <Badge size="xs" color="red" variant="light">오검출 {w.fp}</Badge>}
                        <Text size="xs" c="dimmed" truncate="end">{w.name}</Text>
                      </Group>
                    </Stack>
                  ))}
                </SimpleGrid>
              </Stack>
            )}
          </Stack>
        )}
      </Stack>

      <Modal opened={!!enlarged} onClose={() => setEnlarged(null)} size="xl" title={enlarged?.name}>
        {enlarged && (
          <Stack gap="xs">
            <DiffOverlay src={enlarged.url} gt={enlarged.gt_boxes} pred={enlarged.pred_boxes} />
            <Group gap="xs">
              <Badge color="teal" variant="light">TP {enlarged.tp}</Badge>
              <Badge color="orange" variant="light">놓침(FN) {enlarged.fn}</Badge>
              <Badge color="red" variant="light">오검출(FP) {enlarged.fp}</Badge>
            </Group>
          </Stack>
        )}
      </Modal>
    </Card>
  )
}

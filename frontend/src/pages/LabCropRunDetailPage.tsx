// 크롭 런 상세 — 학습 상세페이지와 같은 구조다.
//
//   상단  설정 표 (쓴 모델 · 추론 방식 · 노브 · 비율 · 출력 옵션)
//   하단  영상 두 갈래 — 왼쪽 가로(wide.mp4), 오른쪽 세로(crop.mp4)
//   그 아래 다운로드
//
// 진행 중이면 같은 자리에 진행률(SSE)이 온다. 전역 잡 카드는 쓰지 않는다 —
// progress.jsonl 을 처음부터 재생하므로 탭을 옮기거나 새로고침해도 이어진다.
import { useEffect, useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Group,
  Progress,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import {
  IconAlertTriangle,
  IconChevronLeft,
  IconCopy,
  IconDownload,
  IconPlayerStop,
} from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  cancelLabCropRun,
  duplicateLabCropRun,
  getLabCropRun,
  labCropJsonUrl,
  labCutVideoUrl,
  labWideVideoUrl,
  subscribeLabCropEvents,
  type LabCropProgressEvent,
} from '../api/client'

const STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  cancelled: 'yellow',
}

// 워커가 내는 페이즈 → 사람이 읽는 말
const PHASE_LABEL: Record<string, string> = {
  start: 'Starting…',
  detect: 'Detecting objects',
  crop_analyze: 'Computing crop path',
  render: 'Rendering videos',
  encoding: 'Encoding',
  done: 'Done',
  error: 'Failed',
  cancelled: 'Cancelled',
}

const TOGGLE_LABEL: Record<string, string> = {
  obj_boxes: 'Detection boxes',
  player_trails: 'Player trails',
  ball_trail: 'Ball trail',
  target_highlight: 'Target highlight',
  crop_box: 'Crop box',
  dead_zone: 'Dead zone',
  center_line: 'Center line',
}

// 지금 상태를 말한다 — 어떻게 만들어졌는지가 아니라. 하드링크로 만든 사본도
// 아카이브에서 원본을 지우면 링크 수가 1 로 떨어져 홀로 남는다(파일은 온전하다).
function percent(ev: LabCropProgressEvent | null): number | null {
  if (!ev?.total || ev.done == null) return null
  return Math.min(100, Math.round((ev.done / ev.total) * 100))
}

export default function LabCropRunDetailPage() {
  const { cropId = '' } = useParams()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [event, setEvent] = useState<LabCropProgressEvent | null>(null)

  const run = useQuery({
    queryKey: ['lab-crop', cropId],
    queryFn: () => getLabCropRun(cropId),
    enabled: !!cropId,
  })

  const running = run.data?.status === 'running'

  // 진행률은 서버가 안다 — 붙을 때마다 처음부터 재생받는다
  useEffect(() => {
    if (!running) return
    const stop = subscribeLabCropEvents(cropId, (ev) => {
      setEvent(ev)
      if (ev.phase === 'done' || ev.phase === 'error' || ev.phase === 'cancelled') {
        qc.invalidateQueries({ queryKey: ['lab-crop', cropId] })
        qc.invalidateQueries({ queryKey: ['lab-crops'] })
      }
    })
    return stop
  }, [running, cropId, qc])

  const cancel = useMutation({
    mutationFn: () => cancelLabCropRun(cropId),
    onSuccess: () => notifications.show({ message: 'Cancelling…', color: 'yellow' }),
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const duplicate = useMutation({
    mutationFn: () => duplicateLabCropRun(cropId),
    onSuccess: (copy) => {
      qc.invalidateQueries({ queryKey: ['lab-crops'] })
      navigate(`/lab/crops/${copy.id}`)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const r = run.data
  if (!r) return null

  const pct = percent(event)
  const enabledToggles = Object.entries(r.toggles).filter(([, on]) => on)
  const overrideRows = Object.entries(r.overrides)

  const rows: [string, React.ReactNode][] = [
    ['Source', r.source_name || '–'],
    [
      'Models',
      r.models.length
        ? r.models
            .map((m) => `${m.name} · ${m.mode}${m.conf != null ? ` · conf ${m.conf}` : ''}`)
            .join('\n')
        : '–',
    ],
    [
      'Crop size',
      r.applied_w && (r.applied_w !== r.crop_w || r.applied_h !== r.crop_h)
        ? `${r.applied_w}×${r.applied_h} px (asked ${r.crop_w}×${r.crop_h}, clamped to the source)`
        : `${r.applied_w || r.crop_w}×${r.applied_h || r.crop_h} px`,
    ],
    [
      'Overlays',
      enabledToggles.length
        ? enabledToggles.map(([k]) => TOGGLE_LABEL[k] ?? k).join(' · ')
        : 'none — wide video is the original',
    ],
    ['Created', r.created_at ? new Date(r.created_at).toLocaleString() : '–'],
  ]

  return (
    <Stack gap="lg">
      <div>
        <Anchor component={Link} to={'/lab/crops'} size="sm" c="dimmed">
          <Group gap={4}>
            <IconChevronLeft size={14} />
            Crop Runs
          </Group>
        </Anchor>
      </div>

      <Group justify="space-between" align="flex-start">
        <div>
          <Group gap="sm">
            <Title order={3}>{r.name}</Title>
            <Badge variant="light" color={STATUS_COLOR[r.status] ?? 'gray'}>
              {r.status}
            </Badge>
          </Group>
        </div>
        <Group gap="xs">
          {running && (
            <Button
              variant="light"
              color="yellow"
              leftSection={<IconPlayerStop size={16} />}
              loading={cancel.isPending}
              onClick={() => cancel.mutate()}
            >
              Cancel
            </Button>
          )}
          <Tooltip label="Same settings, run again — swap the models to compare">
            <Button
              variant="light"
              leftSection={<IconCopy size={16} />}
              loading={duplicate.isPending}
              onClick={() => duplicate.mutate()}
            >
              Duplicate
            </Button>
          </Tooltip>
        </Group>
      </Group>

      {r.error && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Run failed">
          {r.error}
        </Alert>
      )}

      {running && (
        <Card withBorder radius="md" padding="md">
          <Group justify="space-between" mb={6}>
            <Text size="sm" fw={600}>
              {PHASE_LABEL[event?.phase ?? 'start'] ?? event?.phase}
            </Text>
            {pct != null && (
              <Text size="sm" c="dimmed">
                {pct}%
              </Text>
            )}
          </Group>
          <Progress value={pct ?? 100} animated striped={pct == null} />
        </Card>
      )}

      <Card withBorder radius="md" padding="md">
        <Text size="sm" fw={600} mb="xs">
          Run details
        </Text>
        <Table variant="vertical" withRowBorders={false} verticalSpacing={4}>
          <Table.Tbody>
            {rows.map(([k, v]) => (
              <Table.Tr key={k}>
                <Table.Th
                  w={160}
                  style={{ color: 'var(--mantine-color-dimmed)', fontWeight: 500 }}
                >
                  {k}
                </Table.Th>
                <Table.Td>
                  <Text size="sm" style={{ whiteSpace: 'pre-line' }}>
                    {v}
                  </Text>
                </Table.Td>
              </Table.Tr>
            ))}
            {overrideRows.length > 0 && (
              <Table.Tr>
                <Table.Th
                  w={160}
                  style={{ color: 'var(--mantine-color-dimmed)', fontWeight: 500 }}
                >
                  Tuning
                </Table.Th>
                <Table.Td>
                  <Text size="sm">
                    {overrideRows.map(([k, v]) => `${k} = ${v}`).join(' · ')}
                  </Text>
                </Table.Td>
              </Table.Tr>
            )}
            {r.summary && (
              <Table.Tr>
                <Table.Th
                  w={160}
                  style={{ color: 'var(--mantine-color-dimmed)', fontWeight: 500 }}
                >
                  Coverage
                </Table.Th>
                <Table.Td>
                  <Text size="sm">
                    ball {((r.summary.ballTrackingCoverage ?? 0) * 100).toFixed(0)}% · player{' '}
                    {((r.summary.playerTrackingCoverage ?? 0) * 100).toFixed(0)}% · fallback{' '}
                    {((r.summary.fallbackCoverage ?? 0) * 100).toFixed(0)}% ·{' '}
                    {r.summary.keyframeCount ?? 0} keyframes
                  </Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </Card>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
        <Card withBorder radius="md" padding="md">
          <Group justify="space-between" mb="xs">
            <Text size="sm" fw={600}>
              Wide
            </Text>
          </Group>
          {r.has_wide ? (
            <video
              key={labWideVideoUrl(cropId)}
              src={labWideVideoUrl(cropId)}
              controls
              style={{ width: '100%', borderRadius: 6, background: '#000' }}
            />
          ) : (
            <Text size="sm" c="dimmed">
              {running ? 'Rendering…' : 'Not available.'}
            </Text>
          )}
        </Card>

        <Card withBorder radius="md" padding="md">
          <Group justify="space-between" mb="xs">
            <Text size="sm" fw={600}>
              Crop
            </Text>
            <Text size="xs" c="dimmed">
              {r.applied_w || r.crop_w}×{r.applied_h || r.crop_h} cut
            </Text>
          </Group>
          {r.has_cut ? (
            <video
              key={labCutVideoUrl(cropId)}
              src={labCutVideoUrl(cropId)}
              controls
              style={{
                width: '100%',
                maxHeight: 520,
                objectFit: 'contain',
                borderRadius: 6,
                background: '#000',
              }}
            />
          ) : (
            <Text size="sm" c="dimmed">
              {running ? 'Rendering…' : 'Not available.'}
            </Text>
          )}
        </Card>
      </SimpleGrid>

      <Card withBorder radius="md" padding="md">
        <Group gap="sm">
          <Button
            variant="light"
            component="a"
            href={labCropJsonUrl(cropId)}
            leftSection={<IconDownload size={16} />}
            disabled={!r.has_crop_json}
          >
            crop.json
          </Button>
          <Button
            variant="light"
            component="a"
            href={labWideVideoUrl(cropId)}
            leftSection={<IconDownload size={16} />}
            disabled={!r.has_wide}
          >
            wide.mp4
          </Button>
          <Button
            variant="light"
            component="a"
            href={labCutVideoUrl(cropId)}
            leftSection={<IconDownload size={16} />}
            disabled={!r.has_cut}
          >
            crop.mp4
          </Button>
        </Group>
      </Card>
    </Stack>
  )
}

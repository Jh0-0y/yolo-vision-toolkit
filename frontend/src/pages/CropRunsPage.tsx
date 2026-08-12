import { useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Code,
  Group,
  Modal,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import { IconDownload, IconMovie, IconPencil, IconTrash } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import {
  cropJsonUrl,
  cropVideoUrl,
  deleteCropRun,
  listCropRuns,
  renameCropRun,
  type CropRunOut,
} from '../api/client'

const STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  cancelled: 'gray',
}

function pct(v: number | undefined): string {
  return v == null ? '—' : `${Math.round(v * 100)}%`
}

/** 튜닝 노브 요약 — 기본값에서 벗어난 것만 보여준다. 어떤 설정이 어떤 커버리지를
 *  냈는지 나란히 읽는 것이 이 목록의 쓸모다. */
function knobs(run: CropRunOut): string {
  const overrides = run.settings.overrides
  if (!overrides || typeof overrides !== 'object') return '—'
  const entries = Object.entries(overrides as Record<string, unknown>)
  if (!entries.length) return 'defaults'
  return entries.map(([k, v]) => `${k}=${v}`).join(' · ')
}

export default function CropRunsPage() {
  const { projectId = '' } = useParams()
  const qc = useQueryClient()
  const [renaming, setRenaming] = useState<CropRunOut | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const runs = useQuery({
    queryKey: ['crop-runs', projectId],
    queryFn: () => listCropRuns(projectId),
    // 도는 런이 있으면 상태가 서버에서 바뀐다 — 목록 자체가 진행 현황이다.
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === 'running') ? 3000 : false),
  })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['crop-runs', projectId] })

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameCropRun(projectId, id, name),
    onSuccess: (r) => {
      notifications.show({ message: `Renamed: ${r.name}`, color: 'green' })
      setRenaming(null)
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => deleteCropRun(projectId, id),
    onSuccess: () => {
      notifications.show({ message: 'Crop run deleted', color: 'green' })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const data = runs.data ?? []

  return (
    <Stack gap="lg">
      <div>
        <Title order={3}>Crop Runs</Title>
        <Text c="dimmed" size="sm">
          Every crop job you started in this project, with the settings it ran under. Coordinates
          stay; the rendered video is temporary and expires after an hour.
        </Text>
      </div>

      <Card withBorder radius="md" padding="sm">
        <Table highlightOnHover verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Name</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Coverage</Table.Th>
              <Table.Th>Tuning</Table.Th>
              <Table.Th>Created</Table.Th>
              <Table.Th w={130} />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {data.map((r) => (
              <Table.Tr key={r.id}>
                <Table.Td>
                  <Text size="sm" fw={600}>{r.name}</Text>
                  <Text size="xs" c="dimmed">
                    {r.kind === 'cut' ? 'crop cut' : 'coordinates'} · {r.source_name}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Badge variant="light" color={STATUS_COLOR[r.status] ?? 'gray'}>
                    {r.status}
                  </Badge>
                  {r.error && (
                    <Text size="xs" c="red" lineClamp={2} maw={220}>
                      {r.error}
                    </Text>
                  )}
                  {r.video_expired && (
                    <Text size="xs" c="dimmed">video expired · JSON only</Text>
                  )}
                </Table.Td>
                <Table.Td>
                  {r.summary ? (
                    <Text size="xs">
                      ball <Code>{pct(r.summary.ballTrackingCoverage)}</Code> · player{' '}
                      <Code>{pct(r.summary.playerTrackingCoverage)}</Code> · fallback{' '}
                      <Code>{pct(r.summary.fallbackCoverage)}</Code> · keyframes{' '}
                      <Code>{r.summary.keyframeCount ?? '—'}</Code>
                    </Text>
                  ) : (
                    <Text size="xs" c="dimmed">—</Text>
                  )}
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed" lineClamp={2} maw={240}>{knobs(r)}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed">{new Date(r.created_at).toLocaleString()}</Text>
                </Table.Td>
                <Table.Td>
                  <Group gap={4} justify="flex-end" wrap="nowrap">
                    <Tooltip label={r.has_crop_json ? 'Download crop.json' : 'No coordinates'}>
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        disabled={!r.has_crop_json}
                        component="a"
                        href={cropJsonUrl(projectId, r.id)}
                        download={`${r.name}.crop.json`}
                      >
                        <IconDownload size={16} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label={r.has_video ? 'Download video' : 'No video'}>
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        disabled={!r.has_video}
                        component="a"
                        href={cropVideoUrl(projectId, r.id)}
                        download={`${r.name}.mp4`}
                      >
                        <IconMovie size={16} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Rename">
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        onClick={() => {
                          setRenaming(r)
                          setRenameValue(r.name)
                        }}
                      >
                        <IconPencil size={16} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Delete">
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        loading={remove.isPending && remove.variables === r.id}
                        onClick={() => {
                          if (confirm(`Delete crop run "${r.name}"? Its files are removed.`))
                            remove.mutate(r.id)
                        }}
                      >
                        <IconTrash size={16} />
                      </ActionIcon>
                    </Tooltip>
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
        {data.length === 0 && (
          <Text size="sm" c="dimmed" p="md">
            No crop runs yet. Start one from the Crop page.
          </Text>
        )}
      </Card>

      <Modal opened={renaming !== null} onClose={() => setRenaming(null)} title="Rename crop run" size="sm">
        <form
          onSubmit={(ev) => {
            ev.preventDefault()
            const name = renameValue.trim()
            if (renaming && name && !rename.isPending) rename.mutate({ id: renaming.id, name })
          }}
        >
          <Stack>
            <TextInput
              label="Run name"
              value={renameValue}
              onChange={(ev) => setRenameValue(ev.currentTarget.value)}
              data-autofocus
            />
            <Button type="submit" disabled={!renameValue.trim()} loading={rename.isPending}>
              Save
            </Button>
          </Stack>
        </form>
      </Modal>
    </Stack>
  )
}

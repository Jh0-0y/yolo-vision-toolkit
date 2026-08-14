// 크롭 이력 — 어떤 설정으로 무엇을 돌렸는지가 남는 자리.
//
// 실패한 시도도 남는다. 어떤 설정이 안 되는지가 연구실에서는 값지다.
// 도는 게 있으면 3초로 당겨 폴링한다(TrainingHistoryPage 와 같은 패턴).
import { useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Modal,
  SegmentedControl,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconPencil, IconPlus, IconTrash } from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import {
  deleteLabCropRun,
  listLabCropRuns,
  renameLabCropRun,
  type LabCropRunOut,
} from '../api/client'

const STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  cancelled: 'yellow',
}

const FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'running', label: 'Running' },
  { value: 'done', label: 'Done' },
  { value: 'error', label: 'Failed' },
]

function formatBytes(n: number): string {
  if (!n) return '–'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

export default function LabCropRunsPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [filter, setFilter] = useState('all')
  const [renaming, setRenaming] = useState<LabCropRunOut | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const runs = useQuery({
    queryKey: ['lab-crops'],
    queryFn: () => listLabCropRuns(),
    // 도는 게 있으면 빨리, 없으면 느리게
    refetchInterval: (query) => {
      const data = query.state.data as LabCropRunOut[] | undefined
      return data?.some((r) => r.status === 'running') ? 3_000 : 15_000
    },
  })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['lab-crops'] })
    qc.invalidateQueries({ queryKey: ['lab'] })
  }

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameLabCropRun(id, name),
    onSuccess: () => {
      setRenaming(null)
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => deleteLabCropRun(id),
    onSuccess: () => {
      notifications.show({ message: 'Crop run deleted', color: 'green' })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const all = runs.data ?? []
  const data = filter === 'all' ? all : all.filter((r) => r.status === filter)

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={3}>Crop Runs</Title>
          <Text c="dimmed" size="sm">
            Every run keeps its settings, coordinates and both videos. Nothing expires — you delete
            them.
          </Text>
        </div>
        <Button
          leftSection={<IconPlus size={16} />}
          component={Link}
          to={'/lab/crop'}
        >
          New run
        </Button>
      </Group>

      <SegmentedControl value={filter} onChange={setFilter} data={FILTERS} w="fit-content" />

      <Card withBorder radius="md" padding="sm">
        <Table highlightOnHover verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Name</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Models</Table.Th>
              <Table.Th>Crop</Table.Th>
              <Table.Th>Size</Table.Th>
              <Table.Th>Created</Table.Th>
              <Table.Th w={90} />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {data.map((r) => (
              <Table.Tr
                key={r.id}
                style={{ cursor: 'pointer' }}
                onClick={() => navigate(`/lab/crops/${r.id}`)}
              >
                <Table.Td>
                  <Text size="sm" fw={600}>
                    {r.name}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {r.source_name}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Group gap={6} wrap="nowrap">
                    {r.status === 'running' && <Loader size={14} />}
                    <Badge variant="light" color={STATUS_COLOR[r.status] ?? 'gray'}>
                      {r.status}
                    </Badge>
                  </Group>
                  {r.error && (
                    <Text size="xs" c="red" lineClamp={1}>
                      {r.error}
                    </Text>
                  )}
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed" lineClamp={2}>
                    {r.models.map((m) => `${m.name} (${m.mode})`).join(' · ') || '–'}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">
                    {r.applied_w || r.crop_w}×{r.applied_h || r.crop_h}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{formatBytes(r.size_bytes)}</Text>
                  {r.wide_kind === 'link' && (
                    <Text size="xs" c="dimmed" title="가로 영상은 원본 하드링크 — 디스크는 한 벌">
                      linked
                    </Text>
                  )}
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed">
                    {r.created_at ? new Date(r.created_at).toLocaleString() : '–'}
                  </Text>
                </Table.Td>
                <Table.Td onClick={(e) => e.stopPropagation()}>
                  <Group gap={4} justify="flex-end" wrap="nowrap">
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
                          if (confirm(`Delete crop run "${r.name}" and its outputs?`))
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
            {all.length === 0
              ? 'No crop runs yet. Start one from the Crop page.'
              : 'No runs match this filter.'}
          </Text>
        )}
      </Card>

      <Modal
        opened={renaming !== null}
        onClose={() => setRenaming(null)}
        title="Rename crop run"
        size="sm"
      >
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

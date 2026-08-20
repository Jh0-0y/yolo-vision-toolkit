// 데이터셋 목록 — 학습실에 들어오면 처음 보는 화면.
//
// 데이터셋 하나가 실험 하나다("공만" · "선수만"). 카드는 그 데이터셋이 지금 어느
// 단계에 있는지를 보여준다 — 미검수가 얼마나 남았고, 검수된 것이 분할까지 갔는지.
import { useState } from 'react'
import {
  ActionIcon,
  Button,
  Card,
  Group,
  Menu,
  Modal,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconDotsVertical, IconPencil, IconPlus, IconTrash } from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  createDataset,
  deleteDataset,
  listDatasets,
  renameDataset,
  type DatasetOut,
} from '../api/client'
import StatTile from '../components/StatTile'

export default function DatasetsPage() {
  const { projectId = '' } = useParams()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [renaming, setRenaming] = useState<DatasetOut | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const datasets = useQuery({
    queryKey: ['datasets', projectId],
    queryFn: () => listDatasets(projectId),
  })
  const invalidate = () => qc.invalidateQueries({ queryKey: ['datasets', projectId] })

  const create = useMutation({
    mutationFn: (n: string) => createDataset(projectId, n),
    // 만들면 곧장 그 데이터셋으로 — 다음에 할 일은 가져오기다
    onSuccess: (ds) => navigate(`/projects/${projectId}/datasets/${ds.id}`),
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const rename = useMutation({
    mutationFn: ({ id, name: n }: { id: string; name: string }) =>
      renameDataset(projectId, id, n),
    onSuccess: () => {
      setRenaming(null)
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => deleteDataset(projectId, id),
    onSuccess: () => {
      notifications.show({ message: 'Dataset deleted', color: 'green' })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const data = datasets.data ?? []

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={3}>Datasets</Title>
        </div>
        <Button leftSection={<IconPlus size={16} />} onClick={() => setCreating(true)}>
          New dataset
        </Button>
      </Group>

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>
        {data.map((ds) => (
          <Card
            key={ds.id}
            withBorder
            radius="md"
            padding="lg"
            component={Link}
            to={`/projects/${projectId}/datasets/${ds.id}`}
            style={{ textDecoration: 'none' }}
          >
            <Group justify="space-between" align="flex-start" wrap="nowrap">
              <div style={{ minWidth: 0 }}>
                <Text fw={700} size="lg" truncate>
                  {ds.name}
                </Text>
                <Text size="xs" c="dimmed">
                  {ds.created_at ? new Date(ds.created_at).toLocaleString() : '–'}
                </Text>
              </div>
              <Menu position="bottom-end" withinPortal>
                <Menu.Target>
                  <ActionIcon variant="subtle" color="gray" onClick={(e) => e.preventDefault()}>
                    <IconDotsVertical size={16} />
                  </ActionIcon>
                </Menu.Target>
                <Menu.Dropdown>
                  <Menu.Item
                    leftSection={<IconPencil size={14} />}
                    onClick={(e) => {
                      e.preventDefault()
                      setRenaming(ds)
                      setRenameValue(ds.name)
                    }}
                  >
                    Rename
                  </Menu.Item>
                  <Menu.Item
                    color="red"
                    leftSection={<IconTrash size={14} />}
                    onClick={(e) => {
                      e.preventDefault()
                      // 데이터셋 밖으로 공유하는 것이 없으니 통째로 사라진다
                      if (
                        confirm(
                          `Delete dataset "${ds.name}"? Its ${ds.images} images and labels go too.`,
                        )
                      )
                        remove.mutate(ds.id)
                    }}
                  >
                    Delete dataset
                  </Menu.Item>
                </Menu.Dropdown>
              </Menu>
            </Group>

            {/* 미검수 → 검수완료 가 이 화면의 1급 구분이다 */}
            <Group gap="sm" mt="md" grow>
              <StatTile label="Unreviewed" value={ds.unreviewed} color="orange" />
              <StatTile label="Reviewed" value={ds.reviewed} color="teal" />
            </Group>
            <Text size="xs" c="dimmed" mt="xs">
              {ds.unassigned} unassigned · train {ds.train} · val {ds.val} · test {ds.test}
            </Text>
          </Card>
        ))}
      </SimpleGrid>

      {data.length === 0 && (
        <Card withBorder radius="md" padding="xl">
          <Stack align="center" gap="xs">
            <IconPlus size={36} stroke={1.2} />
            <Text c="dimmed">No datasets yet. Create one, then import images into it.</Text>
            <Button variant="light" onClick={() => setCreating(true)}>
              Create a dataset
            </Button>
          </Stack>
        </Card>
      )}

      <Modal opened={creating} onClose={() => setCreating(false)} title="New dataset" size="sm">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const n = name.trim()
            if (n && !create.isPending) create.mutate(n)
          }}
        >
          <Stack>
            <TextInput
              label="Name"
              placeholder="ball only"
              value={name}
              onChange={(e) => setName(e.currentTarget.value)}
              data-autofocus
            />
            <Button type="submit" disabled={!name.trim()} loading={create.isPending}>
              Create
            </Button>
          </Stack>
        </form>
      </Modal>

      <Modal
        opened={renaming !== null}
        onClose={() => setRenaming(null)}
        title="Rename dataset"
        size="sm"
      >
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const n = renameValue.trim()
            if (renaming && n && !rename.isPending) rename.mutate({ id: renaming.id, name: n })
          }}
        >
          <Stack>
            <TextInput
              label="Dataset name"
              value={renameValue}
              onChange={(e) => setRenameValue(e.currentTarget.value)}
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

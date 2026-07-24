import { useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  Modal,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import { IconDownload, IconPencil, IconTrash } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import {
  deleteExport,
  exportDownloadUrl,
  listExports,
  renameExport,
  type ExportOut,
} from '../api/client'

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

export default function ExportsPage() {
  const { projectId = '' } = useParams()
  const qc = useQueryClient()
  const [renaming, setRenaming] = useState<ExportOut | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const exports = useQuery({
    queryKey: ['exports', projectId],
    queryFn: () => listExports(projectId),
  })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['exports', projectId] })
    qc.invalidateQueries({ queryKey: ['train-datasets'] })
  }

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameExport(projectId, id, name),
    onSuccess: (e) => {
      notifications.show({ message: `Renamed: ${e.name}`, color: 'green' })
      setRenaming(null)
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => deleteExport(projectId, id),
    onSuccess: () => {
      notifications.show({ message: 'Export deleted', color: 'green' })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const data = exports.data ?? []

  return (
    <Stack gap="lg">
      <div>
        <Title order={3}>Exports</Title>
        <Text c="dimmed" size="sm">
          Datasets you exported from this project. Rename, download, or delete them.
        </Text>
      </div>

      <Card withBorder radius="md" padding="sm">
        <Table highlightOnHover verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Name</Table.Th>
              <Table.Th>Kind</Table.Th>
              <Table.Th>Contents</Table.Th>
              <Table.Th>Size</Table.Th>
              <Table.Th>Created</Table.Th>
              <Table.Th w={120} />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {data.map((e) => (
              <Table.Tr key={e.id}>
                <Table.Td>
                  <Text size="sm" fw={600}>
                    {e.name}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Badge variant="light" color={e.kind === 'yolo' ? 'blue' : 'gray'}>
                    {e.kind}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed">
                    {e.kind === 'yolo'
                      ? `train ${e.train} · val ${e.val} · ${e.classes} cls`
                      : `${e.count} images`}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{formatBytes(e.size_bytes)}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed">
                    {new Date(e.created_at).toLocaleString()}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Group gap={4} justify="flex-end" wrap="nowrap">
                    <Tooltip label="Download">
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        component="a"
                        href={exportDownloadUrl(projectId, e.id)}
                      >
                        <IconDownload size={16} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Rename">
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        onClick={() => {
                          setRenaming(e)
                          setRenameValue(e.name)
                        }}
                      >
                        <IconPencil size={16} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Delete">
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        loading={remove.isPending && remove.variables === e.id}
                        onClick={() => {
                          if (confirm(`Delete export "${e.name}"? The saved zip is removed.`))
                            remove.mutate(e.id)
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
            No exports yet. Create one from the Dataset page (Export).
          </Text>
        )}
      </Card>

      <Modal
        opened={renaming !== null}
        onClose={() => setRenaming(null)}
        title="Rename export"
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
              label="Export name"
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

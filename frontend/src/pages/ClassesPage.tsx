import { useState } from 'react'
import {
  ActionIcon,
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
import { IconPencil, IconPlus, IconTrash } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { addClass, deleteClass, listClasses, renameClass, type ProjectClass } from '../api/client'
import { classColor } from '../stores/editorStore'

export default function ClassesPage() {
  const { projectId = '' } = useParams()
  const qc = useQueryClient()
  const [newName, setNewName] = useState('')
  const [renaming, setRenaming] = useState<ProjectClass | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const classes = useQuery({
    queryKey: ['classes', projectId],
    queryFn: () => listClasses(projectId),
  })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['classes', projectId] })
    qc.invalidateQueries({ queryKey: ['stats', projectId] })
    qc.invalidateQueries({ queryKey: ['images', projectId] })
    qc.invalidateQueries({ queryKey: ['labels', projectId] })
  }

  const create = useMutation({
    mutationFn: (name: string) => addClass(projectId, name),
    onSuccess: (c) => {
      notifications.show({ message: `Class added: ${c.name}`, color: 'green' })
      setNewName('')
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: number; name: string }) => renameClass(projectId, id, name),
    onSuccess: (c) => {
      notifications.show({ message: `Renamed: ${c.name}`, color: 'green' })
      setRenaming(null)
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: number) => deleteClass(projectId, id),
    onSuccess: (res) => {
      notifications.show({
        message: `Class deleted${res.removed_boxes ? ` (${res.removed_boxes} boxes removed)` : ''}`,
        color: 'green',
      })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const data = classes.data ?? []

  return (
    <Stack gap="lg">
      <div>
        <Title order={3}>Classes</Title>
        <Text c="dimmed" size="sm">
          Add, rename, or delete label classes for this project.
        </Text>
      </div>

      <Card withBorder radius="md" padding="lg">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const name = newName.trim()
            if (name && !create.isPending) create.mutate(name)
          }}
        >
          <Group align="flex-end" gap="sm">
            <TextInput
              label="New class"
              placeholder="e.g. player"
              value={newName}
              onChange={(e) => setNewName(e.currentTarget.value)}
              style={{ flex: 1, maxWidth: 320 }}
            />
            <Button
              type="submit"
              leftSection={<IconPlus size={16} />}
              disabled={!newName.trim()}
              loading={create.isPending}
            >
              Add
            </Button>
          </Group>
        </form>
      </Card>

      <Card withBorder radius="md" padding="sm">
        <Table highlightOnHover verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th w={60}>ID</Table.Th>
              <Table.Th>Name</Table.Th>
              <Table.Th w={90} ta="right" />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {data.map((c) => (
              <Table.Tr key={c.id}>
                <Table.Td>
                  <Text size="sm" c="dimmed">
                    {c.id}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Group gap="xs" wrap="nowrap">
                    <span
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 3,
                        background: classColor(c.id),
                        flexShrink: 0,
                      }}
                    />
                    <Text size="sm" fw={600}>
                      {c.name}
                    </Text>
                  </Group>
                </Table.Td>
                <Table.Td>
                  <Group gap={4} justify="flex-end" wrap="nowrap">
                    <Tooltip label="Rename">
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        onClick={() => {
                          setRenaming(c)
                          setRenameValue(c.name)
                        }}
                      >
                        <IconPencil size={16} />
                      </ActionIcon>
                    </Tooltip>
                    <Tooltip label="Delete">
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        loading={remove.isPending && remove.variables === c.id}
                        onClick={() => {
                          if (
                            confirm(
                              `Delete class "${c.name}"? Its boxes are removed and later class ids shift down.`,
                            )
                          )
                            remove.mutate(c.id)
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
            No classes yet. Add one above, or they appear after auto-labeling / dataset import.
          </Text>
        )}
      </Card>

      <Modal opened={renaming !== null} onClose={() => setRenaming(null)} title="Rename class" size="sm">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const name = renameValue.trim()
            if (renaming && name && !rename.isPending) rename.mutate({ id: renaming.id, name })
          }}
        >
          <Stack>
            <TextInput
              label="Class name"
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

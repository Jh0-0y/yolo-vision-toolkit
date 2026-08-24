// 이 데이터셋의 라벨 클래스.
//
// 클래스는 **데이터셋마다 따로**다 — 자리도 그 데이터셋의 `classes.json` 이다.
// 프로젝트 하나가 서로 다른 클래스 묶음의 데이터셋을 여럿 가질 수 있어야 하기 때문이다.
//
// 삭제는 비가역이다. 그 클래스의 박스를 버리고 **뒤 번호를 하나씩 당긴다** — 0 을
// 지우면 1 이 0 이 된다. 그래서 누르기 전에 몇 개가 사라지고 남은 것이 몇 번이 되는지
// 먼저 보여 준다.
import { useState } from 'react'
import {
  ActionIcon,
  Alert,
  Button,
  Group,
  Loader,
  Modal,
  Stack,
  Table,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core'
import { IconAlertTriangle, IconCheck, IconPencil, IconPlus, IconTrash, IconX } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  addDatasetClass,
  deleteDatasetClass,
  listDatasetClasses,
  renameDatasetClass,
  type DatasetClass,
} from '../../api/client'
import { classColor } from '../../stores/editorStore'

interface Props {
  projectId: string
  datasetId: string
  opened: boolean
  onClose: () => void
}

const swatch = (id: number) => (
  <span
    style={{
      width: 12,
      height: 12,
      borderRadius: 3,
      background: classColor(id),
      flexShrink: 0,
      display: 'inline-block',
    }}
  />
)

/** 이 클래스를 지우면 남는 것들이 몇 번이 되는지 — 번호가 밀리는 것이 놀라움의 원천이다. */
function shiftNote(classes: DatasetClass[], target: DatasetClass): string {
  const after = classes.filter((c) => c.id > target.id)
  if (after.length === 0) return ''
  const shown = after.slice(0, 3)
  const parts = shown.map((c) => `"${c.name}" ${c.id} → ${c.id - 1}`)
  if (after.length > shown.length) parts.push(`and ${after.length - shown.length} more`)
  return parts.join(', ')
}

export default function ClassesModal({ projectId, datasetId, opened, onClose }: Props) {
  const qc = useQueryClient()
  const [newName, setNewName] = useState('')
  const [renaming, setRenaming] = useState<DatasetClass | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [confirming, setConfirming] = useState<DatasetClass | null>(null)

  const classes = useQuery({
    queryKey: ['dataset-classes', projectId, datasetId],
    queryFn: () => listDatasetClasses(projectId, datasetId),
    enabled: opened,
  })

  // 삭제는 뒤 번호를 당기므로 라벨의 클래스 id 가 통째로 바뀐다 — 이 데이터셋을
  // 보고 있는 쿼리를 모두 무효화한다. 하나라도 빠뜨리면 옛 번호가 화면에 남는다.
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['dataset-classes', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['dataset', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['dataset-images', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['image-names', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['labels', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['stats', projectId] })
  }

  const fail = (e: unknown) => notifications.show({ message: String(e), color: 'red' })

  const create = useMutation({
    mutationFn: (name: string) => addDatasetClass(projectId, datasetId, name),
    onSuccess: (c) => {
      notifications.show({ message: `Class added: ${c.name}`, color: 'green' })
      setNewName('')
      invalidate()
    },
    onError: fail,
  })

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: number; name: string }) =>
      renameDatasetClass(projectId, datasetId, id, name),
    onSuccess: (c) => {
      notifications.show({ message: `Renamed: ${c.name}`, color: 'green' })
      setRenaming(null)
      invalidate()
    },
    onError: fail,
  })

  const remove = useMutation({
    mutationFn: (id: number) => deleteDatasetClass(projectId, datasetId, id),
    onSuccess: (res) => {
      notifications.show({
        message: `Class deleted${res.removed_boxes ? ` (${res.removed_boxes} boxes removed)` : ''}`,
        color: 'green',
      })
      setConfirming(null)
      invalidate()
    },
    onError: fail,
  })

  const rows = classes.data ?? []
  const busy = create.isPending || rename.isPending || remove.isPending

  return (
    <>
      <Modal opened={opened} onClose={onClose} title="Classes" size="lg">
        <Stack gap="md">
          <Text c="dimmed" size="sm">
            Label classes for this dataset. Other datasets keep their own.
          </Text>

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
                style={{ flex: 1 }}
              />
              <Button
                type="submit"
                leftSection={<IconPlus size={16} />}
                disabled={!newName.trim() || busy}
                loading={create.isPending}
              >
                Add
              </Button>
            </Group>
          </form>

          {classes.isLoading ? (
            <Group justify="center" py="xl">
              <Loader size="sm" />
            </Group>
          ) : rows.length === 0 ? (
            <Text c="dimmed" size="sm" ta="center" py="lg">
              No classes yet. Add one above, or import a labelled dataset.
            </Text>
          ) : (
            <Table highlightOnHover verticalSpacing="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th w={50}>ID</Table.Th>
                  <Table.Th>Name</Table.Th>
                  <Table.Th w={110} ta="right">
                    Boxes
                  </Table.Th>
                  <Table.Th w={90} />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {rows.map((c) => (
                  <Table.Tr key={c.id}>
                    <Table.Td>
                      <Text size="sm" c="dimmed">
                        {c.id}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      {renaming?.id === c.id ? (
                        <Group gap={4} wrap="nowrap">
                          <TextInput
                            size="xs"
                            value={renameValue}
                            autoFocus
                            onChange={(e) => setRenameValue(e.currentTarget.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Escape') setRenaming(null)
                              if (e.key === 'Enter' && renameValue.trim())
                                rename.mutate({ id: c.id, name: renameValue.trim() })
                            }}
                            style={{ flex: 1 }}
                          />
                          <ActionIcon
                            variant="subtle"
                            color="teal"
                            disabled={!renameValue.trim() || rename.isPending}
                            onClick={() => rename.mutate({ id: c.id, name: renameValue.trim() })}
                          >
                            <IconCheck size={16} />
                          </ActionIcon>
                          <ActionIcon variant="subtle" color="gray" onClick={() => setRenaming(null)}>
                            <IconX size={16} />
                          </ActionIcon>
                        </Group>
                      ) : (
                        <Group gap="xs" wrap="nowrap">
                          {swatch(c.id)}
                          <Text size="sm" fw={600}>
                            {c.name}
                          </Text>
                        </Group>
                      )}
                    </Table.Td>
                    <Table.Td ta="right">
                      <Text size="sm" c={c.boxes === 0 ? 'dimmed' : undefined}>
                        {c.boxes.toLocaleString()}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Group gap={4} justify="flex-end" wrap="nowrap">
                        <Tooltip label="Rename">
                          <ActionIcon
                            variant="subtle"
                            color="gray"
                            disabled={busy}
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
                            disabled={busy}
                            onClick={() => setConfirming(c)}
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
          )}
        </Stack>
      </Modal>

      <Modal
        opened={confirming !== null}
        onClose={() => setConfirming(null)}
        title={confirming ? `Delete class "${confirming.name}"?` : ''}
        size="md"
      >
        {confirming && (
          <Stack gap="md">
            <Alert color="red" icon={<IconAlertTriangle size={16} />}>
              <Stack gap={4}>
                <Text size="sm">
                  {confirming.boxes === 0
                    ? 'No boxes use this class.'
                    : `${confirming.boxes.toLocaleString()} box${confirming.boxes === 1 ? '' : 'es'} will be removed from this dataset's labels.`}
                </Text>
                {shiftNote(rows, confirming) && (
                  <Text size="sm">Later classes shift down: {shiftNote(rows, confirming)}.</Text>
                )}
                <Text size="sm" fw={600}>
                  This cannot be undone.
                </Text>
              </Stack>
            </Alert>
            <Group justify="flex-end" gap="sm">
              <Button variant="default" onClick={() => setConfirming(null)}>
                Cancel
              </Button>
              <Button
                color="red"
                loading={remove.isPending}
                onClick={() => remove.mutate(confirming.id)}
              >
                Delete
              </Button>
            </Group>
          </Stack>
        )}
      </Modal>
    </>
  )
}

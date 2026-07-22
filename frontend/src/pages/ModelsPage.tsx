import { useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  FileButton,
  Group,
  Menu,
  Modal,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import {
  IconCloudDownload,
  IconDotsVertical,
  IconPencil,
  IconTrash,
  IconUpload,
} from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, patchModel, type ModelOut, type OfficialModel } from '../api/client'

const SOURCE_LABEL: Record<string, { label: string; color: string }> = {
  official: { label: 'Official', color: 'blue' },
  upload: { label: 'Uploaded', color: 'grape' },
  trained: { label: 'Trained', color: 'teal' },
}

export default function ModelsPage() {
  const queryClient = useQueryClient()
  const [downloadOpen, setDownloadOpen] = useState(false)
  const [officialName, setOfficialName] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<ModelOut | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const models = useQuery({
    queryKey: ['models'],
    queryFn: () => api.get<ModelOut[]>('/models'),
  })
  const catalog = useQuery({
    queryKey: ['models-official'],
    queryFn: () => api.get<OfficialModel[]>('/models/official'),
    enabled: downloadOpen,
  })

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['models'] })

  const download = useMutation({
    mutationFn: (name: string) => api.post<ModelOut>('/models/official', { name }),
    onSuccess: (m) => {
      notifications.show({
        message: `${m.name} downloaded (${Object.keys(m.classes).length} classes)`,
        color: 'green',
      })
      invalidate()
      setDownloadOpen(false)
      setOfficialName(null)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const upload = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData()
      form.append('file', file)
      return api.upload<ModelOut>('/models', form)
    },
    onSuccess: (m) => {
      notifications.show({ message: `${m.name} uploaded`, color: 'green' })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/models/${id}`),
    onSuccess: invalidate,
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => patchModel(id, name),
    onSuccess: (m) => {
      notifications.show({ message: `Renamed: ${m.name}`, color: 'green' })
      invalidate()
      setRenaming(null)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const defaultOfficial = catalog.data?.find((c) => c.default)?.name ?? 'yolo26n'

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={3}>Models</Title>
          <Text c="dimmed" size="sm">
            Model registry used for auto labeling and training.
          </Text>
        </div>
        <Group gap="xs">
          <Button
            leftSection={<IconCloudDownload size={16} />}
            variant="light"
            onClick={() => setDownloadOpen(true)}
          >
            Download official model
          </Button>
          <FileButton onChange={(f) => f && upload.mutate(f)} accept=".pt">
            {(props) => (
              <Button
                {...props}
                variant="light"
                color="grape"
                leftSection={<IconUpload size={16} />}
                loading={upload.isPending}
              >
                Upload .pt
              </Button>
            )}
          </FileButton>
        </Group>
      </Group>

      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Name</Table.Th>
            <Table.Th>Source</Table.Th>
            <Table.Th>Classes</Table.Th>
            <Table.Th>Registered</Table.Th>
            <Table.Th w={40} />
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {models.data?.map((m) => {
            const names = Object.values(m.classes)
            const source = SOURCE_LABEL[m.source] ?? SOURCE_LABEL.upload
            return (
              <Table.Tr key={m.id}>
                <Table.Td fw={500}>{m.name}</Table.Td>
                <Table.Td>
                  <Badge variant="light" color={source.color}>
                    {source.label}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  <Tooltip label={names.slice(0, 30).join(', ') + (names.length > 30 ? ' …' : '')}>
                    <Text size="sm">{names.length}</Text>
                  </Tooltip>
                </Table.Td>
                <Table.Td>
                  <Text size="sm" c="dimmed">
                    {new Date(m.created_at).toLocaleString()}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Menu position="bottom-end" withinPortal>
                    <Menu.Target>
                      <ActionIcon variant="subtle" color="gray">
                        <IconDotsVertical size={16} />
                      </ActionIcon>
                    </Menu.Target>
                    <Menu.Dropdown>
                      <Menu.Item
                        leftSection={<IconPencil size={14} />}
                        onClick={() => {
                          setRenaming(m)
                          setRenameValue(m.name)
                        }}
                      >
                        Rename
                      </Menu.Item>
                      <Menu.Item
                        color="red"
                        leftSection={<IconTrash size={14} />}
                        onClick={() => {
                          if (confirm(`Delete model "${m.name}"?`)) remove.mutate(m.id)
                        }}
                      >
                        Delete
                      </Menu.Item>
                    </Menu.Dropdown>
                  </Menu>
                </Table.Td>
              </Table.Tr>
            )
          })}
        </Table.Tbody>
      </Table>
      {models.data?.length === 0 && (
        <Text c="dimmed">
          No models registered. Download an official model or upload a trained .pt.
        </Text>
      )}

      <Modal
        opened={downloadOpen}
        onClose={() => setDownloadOpen(false)}
        title="Download official model"
      >
        <Stack>
          <Select
            label="Model"
            placeholder={catalog.isLoading ? 'Loading catalog…' : `Default: ${defaultOfficial}`}
            data={catalog.data?.map((c) => c.name) ?? []}
            value={officialName}
            onChange={setOfficialName}
            searchable
          />
          <Text size="xs" c="dimmed">
            Sizes grow from n to x — larger is more accurate but slower.
          </Text>
          <Button
            onClick={() => download.mutate(officialName ?? defaultOfficial)}
            loading={download.isPending}
            leftSection={<IconCloudDownload size={16} />}
          >
            Download
          </Button>
        </Stack>
      </Modal>

      <Modal opened={renaming != null} onClose={() => setRenaming(null)} title="Rename model">
        <Stack>
          <TextInput
            label="Name"
            value={renameValue}
            onChange={(e) => setRenameValue(e.currentTarget.value)}
            onKeyDown={(e) =>
              e.key === 'Enter' &&
              renameValue.trim() &&
              renaming &&
              rename.mutate({ id: renaming.id, name: renameValue.trim() })
            }
            data-autofocus
          />
          <Button
            onClick={() => renaming && rename.mutate({ id: renaming.id, name: renameValue.trim() })}
            disabled={!renameValue.trim()}
            loading={rename.isPending}
          >
            Save
          </Button>
        </Stack>
      </Modal>
    </Stack>
  )
}

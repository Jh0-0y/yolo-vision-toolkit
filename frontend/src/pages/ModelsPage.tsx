import { useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  Menu,
  Modal,
  Select,
  Stack,
  Table,
  Tabs,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import {
  IconBox,
  IconChartBar,
  IconCloudDownload,
  IconDotsVertical,
  IconDownload,
  IconPencil,
  IconTrash,
} from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api, modelDownloadUrl, patchModel, type ModelOut, type OfficialModel } from '../api/client'
import AddModelMenu from '../components/models/AddModelMenu'
import CompareMode from '../components/test/CompareMode'

const SOURCE_LABEL: Record<string, { label: string; color: string }> = {
  official: { label: 'Official', color: 'blue' },
  upload: { label: 'Uploaded', color: 'grape' },
  trained: { label: 'Trained', color: 'teal' },
}

export default function ModelsPage() {
  const { projectId = '' } = useParams()
  const queryClient = useQueryClient()
  // 탭을 controlled 로 두는 이유는 하나다 — 모델을 들이는 버튼은 Registry 의 액션이라
  // Compare 를 보는 동안에는 헤더에서 빠져야 한다.
  const [tab, setTab] = useState<string | null>('registry')
  const [downloadOpen, setDownloadOpen] = useState(false)
  const [officialName, setOfficialName] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<ModelOut | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const models = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })
  const catalog = useQuery({
    queryKey: ['models-official'],
    queryFn: () => api.get<OfficialModel[]>('/models/official'),
    enabled: downloadOpen,
  })

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['models', projectId] })

  const download = useMutation({
    mutationFn: (name: string) =>
      api.post<ModelOut>('/models/official', { name, project_id: projectId }),
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
      return api.upload<ModelOut>(`/models?project_id=${projectId}`, form)
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
  const rows = models.data ?? []
  // 불러오는 중에는 빈 상태를 띄우지 않는다 — 잠깐 "모델이 없다"가 번쩍이고 사라진다
  const isEmpty = !models.isLoading && rows.length === 0

  const addMenu = (variant?: 'filled' | 'light') => (
    <AddModelMenu
      variant={variant}
      onDownload={() => setDownloadOpen(true)}
      onUpload={(file) => upload.mutate(file)}
      uploading={upload.isPending}
    />
  )

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={3}>Models</Title>
        </div>
        {tab === 'registry' && addMenu()}
      </Group>

      <Tabs value={tab} onChange={setTab}>
        <Tabs.List>
          <Tabs.Tab value="registry" leftSection={<IconBox size={16} />}>
            Registry
          </Tabs.Tab>
          <Tabs.Tab value="compare" leftSection={<IconChartBar size={16} />}>
            Benchmark
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="registry" pt="md">
          <Card withBorder radius="md" padding="sm">
            {isEmpty ? (
              // 모델은 **여기서** 추가할 수 있으니 빈 상태가 안내로 끝나지 않고 손잡이를 준다
              // (학습 이력은 다른 화면에서 시작하므로 거기는 글줄 하나로 둔다).
              <Stack align="center" gap="xs" py="xl">
                <IconBox size={36} stroke={1.2} />
                <Text c="dimmed" size="sm">
                  No models registered yet. Download an official model or upload a trained .pt.
                </Text>
                {addMenu('light')}
              </Stack>
            ) : (
              <Table highlightOnHover verticalSpacing="sm">
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
                  {rows.map((m) => {
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
                          <Tooltip
                            label={
                              names.slice(0, 30).join(', ') + (names.length > 30 ? ' …' : '')
                            }
                          >
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
                                leftSection={<IconDownload size={14} />}
                                component="a"
                                href={modelDownloadUrl(m.id)}
                              >
                                Download .pt
                              </Menu.Item>
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
            )}
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="compare" pt="md">
          <CompareMode projectId={projectId} models={models.data ?? []} />
        </Tabs.Panel>
      </Tabs>

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
          />
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

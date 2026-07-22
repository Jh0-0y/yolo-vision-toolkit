import { useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  FileButton,
  Group,
  Select,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type ModelOut, type OfficialModel } from '../api/client'

export default function ModelsPage() {
  const queryClient = useQueryClient()
  const [officialName, setOfficialName] = useState<string | null>(null)

  const models = useQuery({
    queryKey: ['models'],
    queryFn: () => api.get<ModelOut[]>('/models'),
  })
  const catalog = useQuery({
    queryKey: ['models-official'],
    queryFn: () => api.get<OfficialModel[]>('/models/official'),
  })

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['models'] })

  const download = useMutation({
    mutationFn: (name: string) => api.post<ModelOut>('/models/official', { name }),
    onSuccess: (m) => {
      notifications.show({ message: `${m.name} 다운로드 완료 (${Object.keys(m.classes).length}개 클래스)`, color: 'green' })
      invalidate()
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
      notifications.show({ message: `${m.name} 업로드 완료`, color: 'green' })
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/models/${id}`),
    onSuccess: invalidate,
  })

  const defaultOfficial = catalog.data?.find((c) => c.default)?.name ?? 'yolo26n'

  return (
    <Stack>
      <Title order={3}>모델 레지스트리</Title>

      <Group>
        <Select
          placeholder={`공식 모델 선택 (기본: ${defaultOfficial})`}
          data={catalog.data?.map((c) => c.name) ?? []}
          value={officialName}
          onChange={setOfficialName}
          searchable
          w={260}
        />
        <Button
          onClick={() => download.mutate(officialName ?? defaultOfficial)}
          loading={download.isPending}
        >
          공식 모델 다운로드
        </Button>
        <FileButton onChange={(f) => f && upload.mutate(f)} accept=".pt">
          {(props) => (
            <Button {...props} variant="light" loading={upload.isPending}>
              .pt 업로드
            </Button>
          )}
        </FileButton>
      </Group>

      <Table striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>이름</Table.Th>
            <Table.Th>출처</Table.Th>
            <Table.Th>클래스</Table.Th>
            <Table.Th>등록일</Table.Th>
            <Table.Th />
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {models.data?.map((m) => {
            const names = Object.values(m.classes)
            return (
              <Table.Tr key={m.id}>
                <Table.Td fw={500}>{m.name}</Table.Td>
                <Table.Td>
                  <Badge variant="light" color={m.source === 'official' ? 'blue' : 'grape'}>
                    {m.source === 'official' ? '공식' : '업로드'}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  <Tooltip label={names.slice(0, 30).join(', ') + (names.length > 30 ? ' …' : '')}>
                    <Text size="sm">{names.length}개</Text>
                  </Tooltip>
                </Table.Td>
                <Table.Td>
                  <Text size="sm" c="dimmed">
                    {new Date(m.created_at).toLocaleString()}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <ActionIcon variant="subtle" color="red" onClick={() => remove.mutate(m.id)}>
                    ✕
                  </ActionIcon>
                </Table.Td>
              </Table.Tr>
            )
          })}
        </Table.Tbody>
      </Table>
      {models.data?.length === 0 && (
        <Text c="dimmed">등록된 모델이 없습니다. 공식 모델을 다운로드하거나 학습된 .pt를 업로드하세요.</Text>
      )}
    </Stack>
  )
}

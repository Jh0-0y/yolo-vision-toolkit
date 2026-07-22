import { useState } from 'react'
import {
  ActionIcon,
  Button,
  Card,
  Group,
  Modal,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api, type ProjectOut } from '../api/client'

export default function ProjectsPage() {
  const queryClient = useQueryClient()
  const [opened, setOpened] = useState(false)
  const [name, setName] = useState('')

  const projects = useQuery({
    queryKey: ['projects'],
    queryFn: () => api.get<ProjectOut[]>('/projects'),
  })

  const create = useMutation({
    mutationFn: (n: string) => api.post<ProjectOut>('/projects', { name: n }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setOpened(false)
      setName('')
    },
  })

  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/projects/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }),
  })

  return (
    <Stack>
      <Group justify="space-between">
        <Title order={3}>프로젝트</Title>
        <Button onClick={() => setOpened(true)}>새 프로젝트</Button>
      </Group>

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>
        {projects.data?.map((p) => (
          <Card key={p.id} withBorder component={Link} to={`/projects/${p.id}`} style={{ textDecoration: 'none' }}>
            <Group justify="space-between">
              <div>
                <Text fw={600}>{p.name}</Text>
                <Text size="xs" c="dimmed">
                  {new Date(p.created_at).toLocaleString()}
                </Text>
              </div>
              <ActionIcon
                variant="subtle"
                color="red"
                onClick={(e) => {
                  e.preventDefault()
                  if (confirm(`프로젝트 "${p.name}"와 모든 데이터를 삭제할까요?`)) remove.mutate(p.id)
                }}
              >
                ✕
              </ActionIcon>
            </Group>
          </Card>
        ))}
      </SimpleGrid>
      {projects.data?.length === 0 && <Text c="dimmed">프로젝트가 없습니다. 새 프로젝트를 만들어 시작하세요.</Text>}

      <Modal opened={opened} onClose={() => setOpened(false)} title="새 프로젝트">
        <Stack>
          <TextInput
            label="이름"
            value={name}
            onChange={(e) => setName(e.currentTarget.value)}
            onKeyDown={(e) => e.key === 'Enter' && name.trim() && create.mutate(name.trim())}
            data-autofocus
          />
          <Button onClick={() => create.mutate(name.trim())} disabled={!name.trim()} loading={create.isPending}>
            만들기
          </Button>
        </Stack>
      </Modal>
    </Stack>
  )
}

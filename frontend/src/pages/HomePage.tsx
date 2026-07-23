import { useState } from 'react'
import {
  ActionIcon,
  Button,
  Card,
  Container,
  Group,
  Menu,
  Modal,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
  Title,
} from '@mantine/core'
import {
  IconDotsVertical,
  IconFolderPlus,
  IconLibraryPhoto,
  IconTags,
  IconTool,
  IconTrash,
  IconUserCheck,
} from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api, type ProjectOut, type StatsOut } from '../api/client'
import StatTile from '../components/StatTile'

function ProjectCard({ project, onDelete }: { project: ProjectOut; onDelete: () => void }) {
  const stats = useQuery({
    queryKey: ['stats', project.id],
    queryFn: () => api.get<StatsOut>(`/projects/${project.id}/stats`),
  })
  const s = stats.data

  return (
    <Card
      withBorder
      radius="md"
      padding="lg"
      component={Link}
      to={`/projects/${project.id}`}
      style={{ textDecoration: 'none' }}
    >
      <Group justify="space-between" align="flex-start" wrap="nowrap">
        <div style={{ minWidth: 0 }}>
          <Text fw={700} size="lg" truncate>
            {project.name}
          </Text>
          <Text size="xs" c="dimmed">
            {new Date(project.created_at).toLocaleString()}
          </Text>
        </div>
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <ActionIcon
              variant="subtle"
              color="gray"
              onClick={(e) => e.preventDefault()}
            >
              <IconDotsVertical size={16} />
            </ActionIcon>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Item
              color="red"
              leftSection={<IconTrash size={14} />}
              onClick={(e) => {
                e.preventDefault()
                if (confirm(`Delete project "${project.name}" and all its data?`)) onDelete()
              }}
            >
              Delete project
            </Menu.Item>
          </Menu.Dropdown>
        </Menu>
      </Group>

      <Group gap="sm" mt="md" grow>
        <StatTile
          label="Images"
          value={s?.images ?? '–'}
          color="gray.7"
          icon={<IconLibraryPhoto size={13} />}
        />
        <StatTile
          label="Labeled"
          value={s?.labeled ?? '–'}
          color="blue"
          icon={<IconTags size={13} />}
        />
        <StatTile
          label="Reviewed"
          value={s?.reviewed ?? '–'}
          color="teal"
          icon={<IconUserCheck size={13} />}
        />
      </Group>
    </Card>
  )
}

export default function HomePage() {
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
    <Container size={1200} py="xl">
      <Stack gap="xl">
        <Group justify="space-between" align="flex-end" pt="md">
          <Group gap="md">
            <ThemeIcon size={52} radius="md" variant="light">
              <IconTool size={30} stroke={1.6} />
            </ThemeIcon>
            <div>
              <Title order={2}>YOLO Vision Toolkit</Title>
              <Text c="dimmed" size="sm">
                From data upload to auto labeling and training — pick a project to get started.
              </Text>
            </div>
          </Group>
          <Button leftSection={<IconFolderPlus size={16} />} onClick={() => setOpened(true)}>
            New Project
          </Button>
        </Group>

        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>
          {projects.data?.map((p) => (
            <ProjectCard key={p.id} project={p} onDelete={() => remove.mutate(p.id)} />
          ))}
        </SimpleGrid>

        {projects.data?.length === 0 && (
          <Card withBorder radius="md" padding="xl">
            <Stack align="center" gap="xs">
              <IconFolderPlus size={36} stroke={1.2} />
              <Text c="dimmed">No projects yet. Create one to get started.</Text>
              <Button variant="light" onClick={() => setOpened(true)}>
                Create a project
              </Button>
            </Stack>
          </Card>
        )}

        <Modal opened={opened} onClose={() => setOpened(false)} title="New project">
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
      </Stack>
    </Container>
  )
}

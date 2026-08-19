// 학습실(Studio) 프로젝트 목록. 프로젝트는 **도메인**(농구·야구) 단위다 —
// 어느 클래스를 학습할지는 프로젝트가 아니라 데이터셋이 정한다.
import { useState } from 'react'
import {
  ActionIcon,
  Anchor,
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
  IconChevronLeft,
  IconDatabase,
  IconDotsVertical,
  IconFolderPlus,
  IconLibraryPhoto,
  IconSchool,
  IconTrash,
  IconUserCheck,
} from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api, listDatasets, type ProjectOut } from '../api/client'
import StatTile from '../components/StatTile'

function ProjectCard({ project, onDelete }: { project: ProjectOut; onDelete: () => void }) {
  // 프로젝트는 수치를 갖지 않는다 — 데이터셋들이 갖는다. 여기서 합쳐 보여준다.
  const datasets = useQuery({
    queryKey: ['datasets', project.id],
    queryFn: () => listDatasets(project.id),
  })
  const s = datasets.data?.reduce(
    (acc, d) => ({
      datasets: acc.datasets + 1,
      images: acc.images + d.images,
      reviewed: acc.reviewed + d.reviewed,
    }),
    { datasets: 0, images: 0, reviewed: 0 },
  )

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
            <ActionIcon variant="subtle" color="gray" onClick={(e) => e.preventDefault()}>
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
          label="Datasets"
          value={s?.datasets ?? '–'}
          color="blue"
          icon={<IconDatabase size={13} />}
        />
        <StatTile
          label="Images"
          value={s?.images ?? '–'}
          color="gray.7"
          icon={<IconLibraryPhoto size={13} />}
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

export default function StudioHomePage() {
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
        <div>
          <Anchor component={Link} to="/" size="sm" c="dimmed">
            <Group gap={4}>
              <IconChevronLeft size={14} />
              Home
            </Group>
          </Anchor>
        </div>

        <Group justify="space-between" align="flex-end">
          <Group gap="md">
            <ThemeIcon size={52} radius="md" variant="light" color="blue">
              <IconSchool size={30} stroke={1.6} />
            </ThemeIcon>
            <div>
              <Title order={2}>Studio</Title>
              <Text c="dimmed" size="sm">
                Datasets, labeling and training — one project per domain.
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

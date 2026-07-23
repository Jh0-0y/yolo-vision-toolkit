import {
  AppShell,
  Badge,
  Divider,
  Group,
  NavLink,
  Stack,
  Text,
  Title,
  Tooltip,
  UnstyledButton,
} from '@mantine/core'
import {
  IconBox,
  IconFlask,
  IconFolder,
  IconHistory,
  IconLibraryPhoto,
  IconPlayerPlay,
  IconTag,
  IconTool,
  IconUpload,
} from '@tabler/icons-react'
import { useQuery } from '@tanstack/react-query'
import { Link, Outlet, useLocation, useParams } from 'react-router-dom'
import { api, type DeviceInfo, type Health, type ProjectOut } from '../api/client'

const NAV_SECTIONS = [
  {
    label: 'Data',
    items: [
      { to: 'upload', label: 'Upload Data', icon: IconUpload },
      { to: 'dataset', label: 'Dataset', icon: IconLibraryPhoto },
      { to: 'classes', label: 'Classes', icon: IconTag },
    ],
  },
  {
    label: 'Model',
    items: [
      { to: 'train', label: 'Train', icon: IconPlayerPlay },
      { to: 'history', label: 'Training History', icon: IconHistory },
      { to: 'models', label: 'Models', icon: IconBox },
      { to: 'test', label: 'Test', icon: IconFlask },
    ],
  },
]

export default function ProjectLayout() {
  const { projectId } = useParams()
  const location = useLocation()

  const project = useQuery({
    queryKey: ['project', projectId],
    queryFn: async () => {
      const projects = await api.get<ProjectOut[]>('/projects')
      return projects.find((p) => p.id === projectId) ?? null
    },
  })
  const health = useQuery({
    queryKey: ['health'],
    queryFn: () => api.get<Health>('/health'),
    refetchInterval: 30_000,
  })
  const device = useQuery({
    queryKey: ['device'],
    queryFn: () => api.get<DeviceInfo>('/system/device'),
    staleTime: Infinity,
  })

  const dev = device.data
  const vram =
    dev?.vram_total_mb != null
      ? `${((dev.vram_used_mb ?? 0) / 1024).toFixed(1)} / ${(dev.vram_total_mb / 1024).toFixed(1)} GB`
      : null

  const base = `/projects/${projectId}`

  return (
    <AppShell navbar={{ width: 240, breakpoint: 'sm' }} padding="md">
      <AppShell.Navbar p="xs">
        <AppShell.Section>
          <Stack gap={4} align="center" py="md">
            <UnstyledButton component={Link} to="/">
              <Stack gap={4} align="center">
                <IconTool size={30} stroke={1.6} />
                <Title order={5} ta="center">
                  YOLO Vision Toolkit
                </Title>
              </Stack>
            </UnstyledButton>
          </Stack>
          <Divider mb="xs" />
        </AppShell.Section>

        <AppShell.Section grow>
          {NAV_SECTIONS.map((section) => (
            <Stack key={section.label} gap={2} mb="md">
              <Text size="xs" fw={700} c="dimmed" tt="uppercase" px="sm" mb={2}>
                {section.label}
              </Text>
              {section.items.map((item) => {
                const to = `${base}/${item.to}`
                return (
                  <NavLink
                    key={item.to}
                    component={Link}
                    to={to}
                    label={item.label}
                    leftSection={<item.icon size={18} stroke={1.6} />}
                    active={location.pathname.startsWith(to)}
                    style={{ borderRadius: 6 }}
                  />
                )
              })}
            </Stack>
          ))}
        </AppShell.Section>

        <AppShell.Section>
          <Divider mb="xs" />
          <Stack gap={6} pb={4} align="center">
            {dev && (
              <Tooltip
                multiline
                label={
                  <Stack gap={2}>
                    <Text size="xs">Device: {dev.device_name}</Text>
                    <Text size="xs">Accelerator: {dev.accelerator.toUpperCase()}</Text>
                    {vram && <Text size="xs">VRAM: {vram}</Text>}
                    <Text size="xs">torch {dev.torch_version}</Text>
                    <Text size="xs" c="dimmed">
                      {dev.platform}
                    </Text>
                  </Stack>
                }
              >
                <Badge
                  color={dev.accelerated ? 'teal' : 'orange'}
                  variant="light"
                  style={{ cursor: 'default', maxWidth: 200 }}
                >
                  {dev.accelerated ? `⚡ ${dev.device_label}` : `🐌 ${dev.device_label}`}
                </Badge>
              </Tooltip>
            )}
            <Badge
              color={health.isSuccess ? 'green' : health.isError ? 'red' : 'gray'}
              variant="light"
            >
              {health.isSuccess ? 'API connected' : health.isError ? 'API error' : 'Checking…'}
            </Badge>
          </Stack>
        </AppShell.Section>
      </AppShell.Navbar>

      <AppShell.Main>
        <Group justify="center" align="flex-start">
          <div style={{ width: '100%', maxWidth: 1200 }}>
            <Group gap="xs" pt="md" pb="xl">
              <IconFolder size={22} stroke={1.6} />
              <Title order={3}>{project.data?.name ?? ' '}</Title>
            </Group>
            <Outlet />
          </div>
        </Group>
      </AppShell.Main>
    </AppShell>
  )
}

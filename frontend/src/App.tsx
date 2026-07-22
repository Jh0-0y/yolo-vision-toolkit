import { AppShell, Badge, Group, NavLink, Stack, Text, Title, Tooltip } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { Link, Route, Routes, useLocation } from 'react-router-dom'
import { api, type DeviceInfo, type Health } from './api/client'
import ModelsPage from './pages/ModelsPage'
import ProjectsPage from './pages/ProjectsPage'
import ProjectPage from './pages/ProjectPage'
import ReviewPage from './pages/ReviewPage'
import TrainingPage from './pages/TrainingPage'

const NAV = [
  { to: '/projects', label: '프로젝트 (오토라벨링)' },
  { to: '/models', label: '모델 레지스트리' },
  { to: '/training', label: '학습' },
]

export default function App() {
  const location = useLocation()
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

  return (
    <AppShell header={{ height: 56 }} navbar={{ width: 240, breakpoint: 'sm' }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Title order={4}>YOLO Vision Toolkit</Title>
          <Group gap="xs">
            {dev && (
              <Tooltip
                multiline
                label={
                  <Stack gap={2}>
                    <Text size="xs">디바이스: {dev.device_name}</Text>
                    <Text size="xs">가속: {dev.accelerator.toUpperCase()}</Text>
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
                  style={{ cursor: 'default' }}
                >
                  {dev.accelerated ? `⚡ ${dev.device_label}` : `🐌 ${dev.device_label}`}
                </Badge>
              </Tooltip>
            )}
            <Badge
              color={health.isSuccess ? 'green' : health.isError ? 'red' : 'gray'}
              variant="light"
            >
              {health.isSuccess ? 'API 연결됨' : health.isError ? 'API 오류' : '확인 중'}
            </Badge>
          </Group>
        </Group>
      </AppShell.Header>
      <AppShell.Navbar p="xs">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            component={Link}
            to={item.to}
            label={item.label}
            active={location.pathname.startsWith(item.to)}
          />
        ))}
      </AppShell.Navbar>
      <AppShell.Main>
        <Routes>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:projectId" element={<ProjectPage />} />
          <Route path="/review/:projectId" element={<ReviewPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/training" element={<TrainingPage />} />
        </Routes>
      </AppShell.Main>
    </AppShell>
  )
}

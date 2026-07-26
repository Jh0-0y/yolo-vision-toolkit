import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Group, Stack, Tabs, Text, Title } from '@mantine/core'
import { IconAlertTriangle, IconChartBar, IconVideo } from '@tabler/icons-react'
import { api, getResources, type ModelOut } from '../api/client'
import TrackMode from '../components/test/TrackMode'
import CompareMode from '../components/test/CompareMode'

export default function TestPage() {
  const { projectId = '' } = useParams()

  const modelsQuery = useQuery({
    queryKey: ['models', projectId],
    queryFn: () => api.get<ModelOut[]>(`/models?project_id=${projectId}`),
  })
  const resourcesQuery = useQuery({
    queryKey: ['resources'],
    queryFn: getResources,
    refetchInterval: 5000,
  })

  const models = modelsQuery.data ?? []
  const resources = resourcesQuery.data

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={3}>Test</Title>
          <Text c="dimmed" size="sm">
            Track objects in a video, or compare model accuracy against your labeled data.
          </Text>
        </div>
        {resources && (
          <Badge variant="light" color={resources.training_active ? 'orange' : 'gray'}>
            {resources.device_label}
          </Badge>
        )}
      </Group>

      {resources?.warning && (
        <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="Resource notice">
          {resources.warning}
        </Alert>
      )}

      <Tabs defaultValue="track">
        <Tabs.List>
          <Tabs.Tab value="track" leftSection={<IconVideo size={16} />}>
            Track
          </Tabs.Tab>
          <Tabs.Tab value="compare" leftSection={<IconChartBar size={16} />}>
            Compare
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="track" pt="md">
          <TrackMode projectId={projectId} models={models} />
        </Tabs.Panel>
        <Tabs.Panel value="compare" pt="md">
          <CompareMode projectId={projectId} models={models} />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}

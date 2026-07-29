import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Group, Stack, Text, Title } from '@mantine/core'
import { IconAlertTriangle } from '@tabler/icons-react'
import { api, getResources, type ModelOut } from '../api/client'
import TrackMode from '../components/test/TrackMode'

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
            Track objects in a video with a trained model.
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

      <TrackMode projectId={projectId} models={models} />
    </Stack>
  )
}

import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Group, Stack, Tabs, Text, Title } from '@mantine/core'
import { IconAlertTriangle, IconBrush, IconFileCode, IconMovie } from '@tabler/icons-react'
import { api, getResources, type ModelOut } from '../api/client'
import CropDrawMode from '../components/lab/CropDrawMode'
import CropResultMode from '../components/lab/CropResultMode'
import CropVideoMode from '../components/lab/CropVideoMode'

export default function CropPage() {
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
          <Title order={3}>Crop</Title>
          <Text c="dimmed" size="sm">
            Vertical 9:16 crop — coordinates (crop.json), the cut clip, and the live tuning draw tool.
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

      <Tabs defaultValue="json" keepMounted={false}>
        <Tabs.List>
          <Tabs.Tab value="json" leftSection={<IconFileCode size={16} />}>
            Crop JSON
          </Tabs.Tab>
          <Tabs.Tab value="draw" leftSection={<IconBrush size={16} />}>
            Crop Draw
          </Tabs.Tab>
          <Tabs.Tab value="video" leftSection={<IconMovie size={16} />}>
            Crop Video
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="json" pt="md">
          <CropResultMode projectId={projectId} models={models} />
        </Tabs.Panel>
        <Tabs.Panel value="draw" pt="md">
          <CropDrawMode projectId={projectId} models={models} />
        </Tabs.Panel>
        <Tabs.Panel value="video" pt="md">
          <CropVideoMode projectId={projectId} />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}

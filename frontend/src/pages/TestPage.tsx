import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Grid, Group, SegmentedControl, Stack, Text, Title } from '@mantine/core'
import { IconAlertTriangle, IconFlask } from '@tabler/icons-react'
import { api, getResources, type ModelOut } from '../api/client'
import TestControls, { type TestConfig } from '../components/test/TestControls'
import SingleMode from '../components/test/SingleMode'
import BatchMode from '../components/test/BatchMode'
import ABMode from '../components/test/ABMode'

type Mode = 'single' | 'batch' | 'ab'

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

  const [mode, setMode] = useState<Mode>('single')
  const [cfg, setCfg] = useState<TestConfig>({
    selected: [],
    device: 'auto',
    iou: 0.55,
    imgsz: '640',
    conf: 0.4,
    showLabels: true,
  })
  const set = <K extends keyof TestConfig>(key: K, value: TestConfig[K]) =>
    setCfg((prev) => ({ ...prev, [key]: value }))

  const models = modelsQuery.data ?? []
  const resources = resourcesQuery.data

  useEffect(() => {
    if (models.length && cfg.selected.length === 0) set('selected', [models[0].id])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

  const deviceOptions = useMemo(() => {
    const opts = ['auto', 'cpu']
    const accel = resources?.accelerator
    if (accel === 'mps' || accel === 'cuda') opts.splice(1, 0, accel)
    return opts.map((v) => ({ label: v.toUpperCase(), value: v }))
  }, [resources?.accelerator])

  return (
    <Stack gap="md" mt="md">
      <Group justify="space-between" align="center">
        <Group gap="xs">
          <IconFlask size={22} />
          <Title order={3}>Test</Title>
          <Text c="dimmed" size="sm">
            학습한 모델을 이미지에 바로 돌려보세요 (저장되지 않음)
          </Text>
        </Group>
        {resources && (
          <Badge variant="light" color={resources.training_active ? 'orange' : 'gray'}>
            {resources.device_label}
            {resources.resident_models > 0 ? ` · 상주 ${resources.resident_models}` : ''}
          </Badge>
        )}
      </Group>

      {resources?.warning && (
        <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="리소스 주의">
          {resources.warning}
        </Alert>
      )}

      <SegmentedControl
        value={mode}
        onChange={(v) => setMode(v as Mode)}
        data={[
          { label: '단일 이미지', value: 'single' },
          { label: '배치 그리드', value: 'batch' },
          { label: 'A/B 비교', value: 'ab' },
        ]}
      />

      <Grid gutter="md">
        <Grid.Col span={{ base: 12, md: 3 }}>
          <TestControls
            models={models}
            loading={modelsQuery.isLoading}
            cfg={cfg}
            set={set}
            deviceOptions={deviceOptions}
            hideModelSelect={mode === 'ab'}
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 9 }}>
          {mode === 'single' && <SingleMode projectId={projectId} cfg={cfg} />}
          {mode === 'batch' && <BatchMode projectId={projectId} cfg={cfg} />}
          {mode === 'ab' && <ABMode projectId={projectId} cfg={cfg} models={models} />}
        </Grid.Col>
      </Grid>
    </Stack>
  )
}

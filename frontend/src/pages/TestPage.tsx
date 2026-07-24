import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Alert, Badge, Card, Grid, Group, SegmentedControl, Stack, Text, Title } from '@mantine/core'
import { IconAlertTriangle, IconFlask } from '@tabler/icons-react'
import { api, getResources, type ModelOut } from '../api/client'
import TestControls, { type TestConfig } from '../components/test/TestControls'
import AnnotateMode from '../components/test/AnnotateMode'

type Mode = 'annotate' | 'analyze'

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

  const [mode, setMode] = useState<Mode>('annotate')
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
          <Text c="dimmed" size="sm">학습한 모델을 영상에 돌려보고, 라벨 데이터로 성능을 분석합니다</Text>
        </Group>
        {resources && (
          <Badge variant="light" color={resources.training_active ? 'orange' : 'gray'}>
            {resources.device_label}
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
          { label: '영상 주석', value: 'annotate' },
          { label: '정밀 분석', value: 'analyze' },
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
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 9 }}>
          {mode === 'annotate' && <AnnotateMode projectId={projectId} cfg={cfg} />}
          {mode === 'analyze' && (
            <Card withBorder radius="md" padding="xl">
              <Text c="dimmed" ta="center">정밀 분석 — 준비 중</Text>
            </Card>
          )}
        </Grid.Col>
      </Grid>
    </Stack>
  )
}

// 데이터셋 상세 — 가져오기부터 내보내기까지 한 화면에서 끝난다.
//
// 이미지는 한 방향으로 흐른다.
//
//   가져오기 → 미검수 → (검수) → 검수완료·미할당 → (비율 split) → train / val / test
//
// 그래서 1급 구분은 **미검수 / 검수완료** 이고, 상단 수치도 그 순서로 보인다.
// 분할은 검수완료 안의 하위 구분이다.
//
// 지금은 껍데기다 — 가져오기(2단계) · 검수·오토라벨링(3단계) · 분할·내보내기(4단계)가
// 각자 자기 자리에 붙는다.
import { useState } from 'react'
import { Anchor, Badge, Button, Group, Stack, Tabs, Text, Title } from '@mantine/core'
import { IconChevronLeft, IconChecks, IconHistory, IconInbox } from '@tabler/icons-react'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { getDataset } from '../api/client'
import StatTile from '../components/StatTile'
import ImagePanel from '../components/dataset/ImagePanel'
import SourcesModal from '../components/dataset/SourcesModal'

const TABS = ['unreviewed', 'reviewed']

export default function DatasetDetailPage() {
  const { projectId = '', datasetId = '' } = useParams()
  // 탭을 URL 에 둔다 — 새로고침해도 보던 탭이다
  const [params, setParams] = useSearchParams()
  const tab = TABS.includes(params.get('tab') ?? '') ? (params.get('tab') as string) : 'unreviewed'
  const [sourcesOpen, setSourcesOpen] = useState(false)

  const dataset = useQuery({
    queryKey: ['dataset', projectId, datasetId],
    queryFn: () => getDataset(projectId, datasetId),
    enabled: !!projectId && !!datasetId,
  })

  const ds = dataset.data
  if (!ds) return null

  return (
    <Stack gap="lg">
      <div>
        <Anchor component={Link} to={`/projects/${projectId}/datasets`} size="sm" c="dimmed">
          <Group gap={4}>
            <IconChevronLeft size={14} />
            Datasets
          </Group>
        </Anchor>
      </div>

      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={3}>{ds.name}</Title>
          <Text c="dimmed" size="sm">
            {ds.images} images in this dataset
          </Text>
        </div>
        {/* 출처는 두 탭 어디서든 닿아야 하므로 탭 밖, 제목 옆에 둔다 */}
        <Button
          variant="default"
          leftSection={<IconHistory size={16} />}
          onClick={() => setSourcesOpen(true)}
        >
          Sources
        </Button>
      </Group>

      <SourcesModal
        projectId={projectId}
        datasetId={datasetId}
        opened={sourcesOpen}
        onClose={() => setSourcesOpen(false)}
      />

      {/* 미검수 · 검수완료 가 먼저, 분할은 그 안의 갈래 */}
      <Group gap="sm" grow align="stretch">
        <StatTile label="Unreviewed" value={ds.unreviewed} color="orange" />
        <StatTile label="Reviewed" value={ds.reviewed} color="teal" />
        <StatTile label="Unassigned" value={ds.unassigned} color="gray.6" />
        <StatTile label="Train" value={ds.train} color="blue" />
        <StatTile label="Val" value={ds.val} color="grape" />
        <StatTile label="Test" value={ds.test} color="pink" />
      </Group>

      <Tabs
        value={tab}
        onChange={(v) => setParams(v ? { tab: v } : {}, { replace: true })}
        keepMounted={false}
      >
        <Tabs.List>
          <Tabs.Tab
            value="unreviewed"
            leftSection={<IconInbox size={16} />}
            rightSection={
              <Badge size="sm" variant="light" color="orange">
                {ds.unreviewed}
              </Badge>
            }
          >
            Unreviewed
          </Tabs.Tab>
          <Tabs.Tab
            value="reviewed"
            leftSection={<IconChecks size={16} />}
            rightSection={
              <Badge size="sm" variant="light" color="teal">
                {ds.reviewed}
              </Badge>
            }
          >
            Reviewed
          </Tabs.Tab>
        </Tabs.List>

        {/* 가져오기 · 나누기 · 내보내기는 **모달**이다 — 가끔 하는 일이 매일 보는
            그리드를 아래로 밀지 않도록. 버튼은 각 패널 위에 있다. */}
        <Tabs.Panel value="unreviewed" pt="md">
          <ImagePanel
            projectId={projectId}
            datasetId={datasetId}
            reviewed={false}
            dataset={ds}
          />
        </Tabs.Panel>

        <Tabs.Panel value="reviewed" pt="md">
          <ImagePanel projectId={projectId} datasetId={datasetId} reviewed dataset={ds} />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}

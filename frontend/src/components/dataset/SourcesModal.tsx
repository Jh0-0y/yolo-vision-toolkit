// 이 데이터가 어디서 왔는지.
//
// 영상은 프레임을 뽑고 **버리므로** 이 목록이 유일한 기록이다. 같은 이름이 두 번
// 들어왔을 때 어느 쪽이 `(2)` 인지도 여기서 본다 — 파일명만 보고는 알 수 없다.
import { Badge, Card, Group, Loader, Modal, Stack, Text } from '@mantine/core'
import { IconFileZip, IconMovie } from '@tabler/icons-react'
import { useQuery } from '@tanstack/react-query'
import { listDatasetSources, type DatasetSource } from '../../api/client'

interface Props {
  projectId: string
  datasetId: string
  opened: boolean
  onClose: () => void
}

const STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'teal',
  error: 'red',
  cancelled: 'gray',
}

/** 영상 추출 설정 중 **결과를 바꾼 것만** 한 줄로. 기본값까지 늘어놓으면 안 읽는다. */
function videoParams(s: DatasetSource): string {
  const p = s.params ?? {}
  const parts: string[] = []
  if (p.target_fps != null) parts.push(`${p.target_fps} fps`)
  if (p.dedup) parts.push('dedup')
  if (p.start_sec) parts.push(`from ${p.start_sec}s`)
  if (p.end_sec != null) parts.push(`to ${p.end_sec}s`)
  return parts.join(' · ')
}

function frameCount(s: DatasetSource): string {
  return s.frames == null ? '…' : `${s.frames} frames`
}

function SourceRow({ s }: { s: DatasetSource }) {
  const video = s.kind === 'video'
  return (
    <Card withBorder radius="md" padding="sm">
      <Group justify="space-between" wrap="nowrap" align="flex-start">
        <Group gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
          {video ? <IconMovie size={18} stroke={1.5} /> : <IconFileZip size={18} stroke={1.5} />}
          <div style={{ minWidth: 0 }}>
            <Text size="sm" fw={500} truncate>
              {s.filename}
            </Text>
            <Text size="xs" c="dimmed">
              {video
                ? `${frameCount(s)}${videoParams(s) ? ` · ${videoParams(s)}` : ''}`
                : `${s.images ?? 0} images · ${s.labeled ?? 0} labeled${
                    s.reviewed ? ` · reviewed, ${s.assigned ?? 0} placed` : ' · unreviewed'
                  }`}
            </Text>
            {/* 프레임 이름의 앞부분 — 그리드에서 이 출처를 찾는 검색어이기도 하다 */}
            {s.stem && (
              <Text size="xs" c="dimmed" ff="monospace">
                {s.stem}_00001 …
              </Text>
            )}
          </div>
        </Group>
        <Stack gap={4} align="flex-end">
          <Badge size="sm" variant="light" color={STATUS_COLOR[s.status] ?? 'gray'}>
            {s.status}
          </Badge>
          <Text size="xs" c="dimmed">
            {new Date(s.at).toLocaleString()}
          </Text>
        </Stack>
      </Group>
    </Card>
  )
}

export default function SourcesModal({ projectId, datasetId, opened, onClose }: Props) {
  const sources = useQuery({
    queryKey: ['dataset-sources', projectId, datasetId],
    queryFn: () => listDatasetSources(projectId, datasetId),
    enabled: opened,
    // 추출이 도는 동안은 프레임 수가 아직 없다 — 끝나면 채워진다
    refetchInterval: (q) =>
      (q.state.data ?? []).some((s) => s.status === 'running') ? 3000 : false,
  })

  return (
    <Modal opened={opened} onClose={onClose} title="Sources" size="lg">
      <Stack gap="sm">
        {sources.isLoading ? (
          <Group justify="center" py="lg">
            <Loader />
          </Group>
        ) : (sources.data ?? []).length === 0 ? (
          <Card withBorder radius="md" padding="lg">
            <Text size="sm" c="dimmed" ta="center">
              Nothing imported yet.
            </Text>
          </Card>
        ) : (
          sources.data?.map((s) => <SourceRow key={s.id} s={s} />)
        )}
      </Stack>
    </Modal>
  )
}

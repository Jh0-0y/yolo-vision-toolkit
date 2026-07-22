import { useState } from 'react'
import {
  Badge,
  Button,
  Card,
  FileButton,
  Group,
  Image,
  Pagination,
  SimpleGrid,
  Stack,
  Tabs,
  Text,
  Title,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { api, type ImagePage, type LabelBox, type StatsOut } from '../api/client'
import ExportSection from '../components/exports/ExportSection'
import JobLaunchModal from '../components/jobs/JobLaunchModal'
import VideoUploadModal from '../components/videos/VideoUploadModal'
import LabelEditModal from '../components/editor/LabelEditModal'
import { classColor } from '../stores/editorStore'

const PAGE_SIZE = 60

export default function ProjectPage() {
  const { projectId = '' } = useParams()
  const queryClient = useQueryClient()
  const [bucket, setBucket] = useState<string>('raw')
  const [page, setPage] = useState(1)
  const [jobModalOpen, setJobModalOpen] = useState(false)
  const [videoModalOpen, setVideoModalOpen] = useState(false)
  const [editStem, setEditStem] = useState<string | null>(null)

  const stats = useQuery({
    queryKey: ['stats', projectId],
    queryFn: () => api.get<StatsOut>(`/projects/${projectId}/stats`),
  })

  const images = useQuery({
    queryKey: ['images', projectId, bucket, page],
    queryFn: () =>
      api.get<ImagePage>(`/projects/${projectId}/images?bucket=${bucket}&page=${page}&size=${PAGE_SIZE}`),
  })

  const upload = useMutation({
    mutationFn: (files: File[]) => {
      const form = new FormData()
      files.forEach((f) => form.append('files', f))
      return api.upload<{ added: number; skipped: number }>(`/projects/${projectId}/images`, form)
    },
    onSuccess: (r) => {
      notifications.show({ message: `${r.added}장 추가됨${r.skipped ? `, ${r.skipped}개 건너뜀` : ''}`, color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['images', projectId] })
      queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const totalPages = images.data ? Math.max(1, Math.ceil(images.data.total / PAGE_SIZE)) : 1

  return (
    <Stack>
      <Group justify="space-between">
        <Title order={3}>프로젝트</Title>
        <Group>
          <FileButton onChange={(files) => files.length && upload.mutate(files)} accept="image/*,.zip" multiple>
            {(props) => (
              <Button {...props} variant="light" loading={upload.isPending}>
                이미지/zip 업로드
              </Button>
            )}
          </FileButton>
          <Button variant="light" onClick={() => setVideoModalOpen(true)}>
            동영상 업로드
          </Button>
          <Button onClick={() => setJobModalOpen(true)} disabled={!stats.data?.raw}>
            오토라벨링 실행
          </Button>
          <Button
            component={Link}
            to={`/review/${projectId}`}
            variant="outline"
            disabled={!stats.data?.review_pending}
          >
            리뷰 ({stats.data?.review_pending ?? 0})
          </Button>
        </Group>
      </Group>

      <Group>
        <Card withBorder padding="xs">
          <Text size="xs" c="dimmed">원본</Text>
          <Text fw={700}>{stats.data?.raw ?? '-'}</Text>
        </Card>
        <Card withBorder padding="xs">
          <Text size="xs" c="dimmed">확정</Text>
          <Text fw={700} c="green">{stats.data?.confirmed ?? '-'}</Text>
        </Card>
        <Card withBorder padding="xs">
          <Text size="xs" c="dimmed">리뷰 대기</Text>
          <Text fw={700} c="orange">{stats.data?.review ?? '-'}</Text>
        </Card>
        {stats.data && stats.data.classes.length > 0 && (
          <Card withBorder padding="xs">
            <Text size="xs" c="dimmed">클래스</Text>
            <Text fw={700}>{stats.data.classes.length}</Text>
          </Card>
        )}
      </Group>

      <Tabs value={bucket} onChange={(v) => { setBucket(v ?? 'raw'); setPage(1) }}>
        <Tabs.List>
          <Tabs.Tab value="raw">원본</Tabs.Tab>
          <Tabs.Tab value="confirmed">확정</Tabs.Tab>
          <Tabs.Tab value="review">검수 대기</Tabs.Tab>
        </Tabs.List>
      </Tabs>

      {images.data && images.data.items.length === 0 && (
        <Text c="dimmed" py="lg">
          {bucket === 'raw' ? '이미지를 업로드하세요.' : '아직 데이터가 없습니다.'}
        </Text>
      )}

      <SimpleGrid cols={{ base: 3, sm: 5, lg: 8 }} spacing="xs">
        {images.data?.items.map((img) => {
          const labeled = bucket === 'confirmed' || bucket === 'review'
          const stem = img.name.replace(/\.[^.]+$/, '')
          return (
            <Stack key={img.name} gap={2}>
              {labeled ? (
                <LabeledThumb
                  thumb={img.thumb}
                  boxes={img.boxes ?? []}
                  onClick={() => setEditStem(stem)}
                />
              ) : (
                <Image
                  src={img.thumb}
                  radius="sm"
                  loading="lazy"
                  style={{ aspectRatio: '1', objectFit: 'cover' }}
                />
              )}
              <Text size="xs" c="dimmed" truncate>
                {img.name}
              </Text>
            </Stack>
          )
        })}
      </SimpleGrid>

      {totalPages > 1 && (
        <Group justify="center">
          <Pagination value={page} onChange={setPage} total={totalPages} />
        </Group>
      )}

      <ExportSection projectId={projectId} confirmedCount={stats.data?.confirmed ?? 0} />

      {stats.data && stats.data.classes.length > 0 && (
        <Group gap="xs">
          {stats.data.classes.map((c) => (
            <Badge key={c.id} variant="light">
              {c.id}: {c.name}
            </Badge>
          ))}
        </Group>
      )}

      <JobLaunchModal projectId={projectId} opened={jobModalOpen} onClose={() => setJobModalOpen(false)} />
      <VideoUploadModal projectId={projectId} opened={videoModalOpen} onClose={() => setVideoModalOpen(false)} />
      <LabelEditModal
        projectId={projectId}
        bucket={bucket === 'review' ? 'review' : 'confirmed'}
        stem={editStem}
        onClose={() => setEditStem(null)}
      />
    </Stack>
  )
}

/** Square thumbnail with normalized label boxes overlaid, exactly aligned to the
 *  contained image regardless of aspect ratio. */
function LabeledThumb({
  thumb,
  boxes,
  onClick,
}: {
  thumb: string
  boxes: LabelBox[]
  onClick: () => void
}) {
  const [aspect, setAspect] = useState<number | null>(null)
  // rect of the contained image within the square box, in % (accounts for letterboxing)
  const rect =
    aspect == null
      ? null
      : aspect >= 1
        ? { left: 0, width: 100, top: (100 - 100 / aspect) / 2, height: 100 / aspect }
        : { top: 0, height: 100, left: (100 - 100 * aspect) / 2, width: 100 * aspect }

  return (
    <div
      onClick={onClick}
      style={{
        position: 'relative',
        aspectRatio: '1',
        cursor: 'pointer',
        borderRadius: 6,
        overflow: 'hidden',
        background: '#111',
      }}
    >
      <img
        src={thumb}
        loading="lazy"
        onLoad={(e) => setAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
        style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
      />
      {rect && boxes.length > 0 && (
        <svg
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          style={{
            position: 'absolute',
            left: `${rect.left}%`,
            top: `${rect.top}%`,
            width: `${rect.width}%`,
            height: `${rect.height}%`,
            pointerEvents: 'none',
          }}
        >
          {boxes.map((b, i) => (
            <rect
              key={i}
              x={b.xyxy_n[0]}
              y={b.xyxy_n[1]}
              width={b.xyxy_n[2] - b.xyxy_n[0]}
              height={b.xyxy_n[3] - b.xyxy_n[1]}
              fill="none"
              stroke={classColor(b.cls)}
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      )}
    </div>
  )
}

// 데이터셋 안의 이미지 그리드 — 미검수 탭과 검수완료 탭이 같은 컴포넌트를 쓴다.
//
// 다른 것은 셋뿐이다.
//   미검수   오토라벨링 버튼이 있고, 고른 것을 "검수 완료"로 보낼 수 있다
//   검수완료 split 하위 필터가 있고, 검수를 되돌릴 수 있다
import { useEffect, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Pagination,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconSparkles, IconTrash, IconChecks, IconArrowBackUp } from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  deleteDatasetImages,
  listDatasetImages,
  setImageReviewed,
  type DatasetImageQuery,
} from '../../api/client'
import { useJobStore } from '../../stores/jobStore'
import AutoLabelModal from './AutoLabelModal'
import ImageGrid from './ImageGrid'

const PAGE_SIZE = 60

const SPLIT_FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'none', label: 'Unassigned' },
  { value: 'train', label: 'Train' },
  { value: 'val', label: 'Val' },
  { value: 'test', label: 'Test' },
]

interface Props {
  projectId: string
  datasetId: string
  /** 이 패널이 보여주는 쪽 */
  reviewed: boolean
}

export default function ImagePanel({ projectId, datasetId, reviewed }: Props) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [page, setPage] = useState(1)
  const [q, setQ] = useState('')
  const [split, setSplit] = useState('all')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [autoLabelOpen, setAutoLabelOpen] = useState(false)

  const query: DatasetImageQuery = {
    reviewed,
    page,
    size: PAGE_SIZE,
    ...(q ? { q } : {}),
    ...(reviewed && split !== 'all' ? { split: split as DatasetImageQuery['split'] } : {}),
  }

  const images = useQuery({
    queryKey: ['dataset-images', projectId, datasetId, query],
    queryFn: () => listDatasetImages(projectId, datasetId, query),
  })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['dataset-images', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['dataset', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['datasets', projectId] })
  }

  // 오토라벨링이 끝나면 박스가 생긴다 — 그때 그리드를 다시 읽는다
  const autoLabelDone = useJobStore((s) =>
    Object.values(s.jobs)
      .filter((j) => j.kind === 'autolabel' && j.status === 'done')
      .reduce((max, j) => Math.max(max, j.seq), 0),
  )
  useEffect(() => {
    if (autoLabelDone > 0) invalidate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoLabelDone])

  // `ImageGrid` 의 선택 집합은 **파일명**이다(stem 이 아니다) — 검수 API 는 stem 을
  // 받으므로 여기서 벗겨 준다. 섞어 쓰면 조용히 404 만 난다.
  const selectedStems = () =>
    items.filter((i) => selected.has(i.name)).map((i) => i.stem)

  const review = useMutation({
    mutationFn: async (next: boolean) => {
      const stems = selectedStems()
      for (const stem of stems) {
        await setImageReviewed(projectId, datasetId, stem, next)
      }
      return stems.length
    },
    onSuccess: (n, next) => {
      notifications.show({
        message: next ? `${n} images marked reviewed` : `${n} images sent back`,
        color: 'green',
      })
      setSelected(new Set())
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: () => deleteDatasetImages(projectId, datasetId, [...selected]),
    onSuccess: (r) => {
      notifications.show({ message: `${r.deleted} images deleted`, color: 'green' })
      setSelected(new Set())
      invalidate()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const data = images.data
  const items = data?.items ?? []
  const pages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE_SIZE))
  // 오토라벨링은 파일명을 받는다 — 선택 집합이 이미 파일명이라 그대로 넘긴다
  const selectedNames = [...selected]

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-end">
        <Group gap="xs">
          <TextInput
            placeholder="Search by name"
            value={q}
            onChange={(e) => {
              setQ(e.currentTarget.value)
              setPage(1)
            }}
            w={220}
          />
          {reviewed && (
            <SegmentedControl
              value={split}
              onChange={(v) => {
                setSplit(v)
                setPage(1)
              }}
              data={SPLIT_FILTERS}
              size="xs"
            />
          )}
        </Group>
        <Group gap="xs">
          {!reviewed && (
            <Button
              variant="light"
              leftSection={<IconSparkles size={16} />}
              onClick={() => setAutoLabelOpen(true)}
            >
              Auto-label{selected.size ? ` (${selected.size})` : ''}
            </Button>
          )}
          {selected.size > 0 && (
            <>
              <Button
                leftSection={reviewed ? <IconArrowBackUp size={16} /> : <IconChecks size={16} />}
                loading={review.isPending}
                onClick={() => review.mutate(!reviewed)}
              >
                {reviewed ? `Send back (${selected.size})` : `Mark reviewed (${selected.size})`}
              </Button>
              <Button
                variant="light"
                color="red"
                leftSection={<IconTrash size={16} />}
                loading={remove.isPending}
                onClick={() => {
                  if (confirm(`Delete ${selected.size} images from this dataset?`)) remove.mutate()
                }}
              >
                Delete
              </Button>
            </>
          )}
        </Group>
      </Group>

      {images.isLoading ? (
        <Group justify="center" py="xl">
          <Loader />
        </Group>
      ) : items.length === 0 ? (
        <Card withBorder radius="md" padding="xl">
          <Text size="sm" c="dimmed" ta="center">
            {reviewed
              ? 'Nothing reviewed yet — mark images in the Unreviewed tab.'
              : 'No unreviewed images. Import some above.'}
          </Text>
        </Card>
      ) : (
        <>
          <Group gap="xs">
            <Badge variant="light" color="gray">
              {data?.total} images
            </Badge>
            {selected.size > 0 && (
              <Badge variant="light">{selected.size} selected</Badge>
            )}
          </Group>
          <ImageGrid
            items={items}
            selected={selected}
            onSelectedChange={setSelected}
            onOpen={(item) =>
              navigate(
                `/projects/${projectId}/datasets/${datasetId}/label/${item.stem}`,
              )
            }
          />
          {pages > 1 && (
            <Group justify="center">
              <Pagination value={page} onChange={setPage} total={pages} />
            </Group>
          )}
        </>
      )}

      <AutoLabelModal
        projectId={projectId}
        datasetId={datasetId}
        opened={autoLabelOpen}
        onClose={() => setAutoLabelOpen(false)}
        names={selectedNames.length ? selectedNames : null}
      />
    </Stack>
  )
}

// 데이터셋 안의 이미지 그리드 — 미검수 탭과 검수완료 탭이 같은 컴포넌트를 쓴다.
//
// 다른 것은 셋뿐이다.
//   미검수   오토라벨링 버튼이 있고, 고른 것을 "검수 완료"로 보낼 수 있다
//   검수완료 split 하위 필터가 있고, 검수를 되돌릴 수 있다
import { useEffect, useState } from 'react'
import {
  Badge,
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
import SelectionBar from './SelectionBar'

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

  // 필터에 걸린 **전부**를 고른다 — 보이는 페이지가 아니다. 서버가 이름만 돌려주는
  // `names_only` 를 쓰므로 수천 장이어도 썸네일을 끌어오지 않는다.
  const selectAll = useMutation({
    mutationFn: () => listDatasetImages(projectId, datasetId, { ...query, names_only: true }),
    onSuccess: (r) => setSelected(new Set(r.names ?? [])),
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  // `ImageGrid` 의 선택 집합은 **파일명**이다(stem 이 아니다) — 검수 API 는 stem 을
  // 받으므로 여기서 벗겨 준다. 섞어 쓰면 조용히 404 만 난다.
  //
  // 이름에서 직접 벗긴다. 보이는 페이지(`items`)로 되짚으면 `Select all` 로 고른
  // 다른 페이지의 이미지가 조용히 빠진다.
  const selectedStems = () => [...selected].map((name) => name.replace(/\.[^.]+$/, ''))

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
        <Badge variant="light" color="gray">
          {data?.total ?? 0} images
        </Badge>
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
          <SelectionBar
            reviewed={reviewed}
            selectedCount={selected.size}
            total={data?.total ?? 0}
            selectingAll={selectAll.isPending}
            reviewPending={review.isPending}
            deletePending={remove.isPending}
            onSelectAll={() => selectAll.mutate()}
            onClear={() => setSelected(new Set())}
            onAutoLabel={() => setAutoLabelOpen(true)}
            onReview={() => review.mutate(!reviewed)}
            onDelete={() => {
              if (confirm(`Delete ${selected.size} images from this dataset?`)) remove.mutate()
            }}
          />
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

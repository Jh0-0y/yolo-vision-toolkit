// 데이터셋 안의 이미지 그리드 — 미검수 탭과 검수완료 탭이 같은 컴포넌트를 쓴다.
//
// 다른 것은 셋뿐이다.
//   미검수   가져오기 · 오토라벨링이 있다
//   검수완료 split 하위 필터와 나누기 · 내보내기가 있고, 검수를 되돌릴 수 있다
//
// **일괄 "검수 완료"는 없다.** 검수는 한 장씩 봐야 의미가 있어서, 검수완료로 보내는
// 것은 라벨 에디터에서만 한다. 되돌리는 쪽은 일괄로 둔다 — 안전한 방향이다.
import { useEffect, useState } from 'react'
import { Button, Card, Group, Loader, Pagination, SegmentedControl, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconArrowsSplit, IconGridDots, IconPackageExport, IconPlus } from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  deleteDatasetImages,
  listDatasetClasses,
  listDatasetImages,
  setImageReviewed,
  type DatasetImageQuery,
  type DatasetOut,
} from '../../api/client'
import { useJobStore } from '../../stores/jobStore'
import AutoLabelModal from './AutoLabelModal'
import ExportModal from './ExportModal'
import GridFilters, { type GridFilterState } from './GridFilters'
import {
  editorSearch,
  readFilters,
  readPage,
  readSplit,
  toImageQuery,
  writeGridParams,
} from './gridParams'
import ImageGrid from './ImageGrid'
import ImportModal from './ImportModal'
import SelectionBar from './SelectionBar'
import SplitModal from './SplitModal'
import TilingModal from './TilingModal'

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
  dataset: DatasetOut
}

export default function ImagePanel({ projectId, datasetId, reviewed, dataset }: Props) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  // 필터·정렬·페이지는 **URL 에** 산다. 그래야 라벨 에디터가 같은 목록을 보고
  // (`3 / 128` 과 이전·다음이 좁혀 놓은 것을 따른다), 새로고침·뒤로가기도 견딘다.
  const [searchParams, setSearchParams] = useSearchParams()
  const filters = readFilters(searchParams)
  const split = readSplit(searchParams)
  const page = readPage(searchParams)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [autoLabelOpen, setAutoLabelOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [splitOpen, setSplitOpen] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)
  const [tileOpen, setTileOpen] = useState(false)

  // 에디터도 **같은 조건**으로 목록을 받는다 — 조건이 어긋나면 순서와 총 수가 달라진다
  const query: DatasetImageQuery = {
    ...toImageQuery(filters, reviewed, split),
    page,
    size: PAGE_SIZE,
  }

  const images = useQuery({
    queryKey: ['dataset-images', projectId, datasetId, query],
    queryFn: () => listDatasetImages(projectId, datasetId, query),
  })

  // 클래스 필터의 선택지 — 이 데이터셋의 클래스다
  const classes = useQuery({
    queryKey: ['dataset-classes', projectId, datasetId],
    queryFn: () => listDatasetClasses(projectId, datasetId),
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

  // 필터가 바뀌면 첫 페이지로. 안 그러면 3페이지짜리 결과의 7페이지를 보게 된다.
  // `replace` 인 이유는 검색어 한 글자마다 히스토리가 쌓이면 뒤로가기가 망가지기 때문이다.
  const applyFilters = (next: GridFilterState) => {
    setSearchParams(writeGridParams(searchParams, { filters: next, page: 1 }), {
      replace: true,
    })
  }

  const applySplit = (next: string) =>
    setSearchParams(writeGridParams(searchParams, { split: next, page: 1 }), { replace: true })

  const applyPage = (next: number) =>
    setSearchParams(writeGridParams(searchParams, { page: next }), { replace: true })

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

  const sendBack = useMutation({
    mutationFn: async () => {
      const stems = selectedStems()
      for (const stem of stems) {
        await setImageReviewed(projectId, datasetId, stem, false)
      }
      return stems.length
    },
    onSuccess: (n) => {
      notifications.show({ message: `${n} images sent back`, color: 'green' })
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

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-end">
        <GridFilters value={filters} onChange={applyFilters} classes={classes.data ?? []} />
        <Group gap="xs">
          {reviewed ? (
            <>
              <Button
                variant="light"
                leftSection={<IconGridDots size={16} />}
                onClick={() => setTileOpen(true)}
              >
                Tile
              </Button>
              <Button
                variant="light"
                leftSection={<IconArrowsSplit size={16} />}
                onClick={() => setSplitOpen(true)}
              >
                Split
              </Button>
              <Button
                variant="light"
                leftSection={<IconPackageExport size={16} />}
                onClick={() => setExportOpen(true)}
              >
                Export
              </Button>
            </>
          ) : (
            <Button leftSection={<IconPlus size={16} />} onClick={() => setImportOpen(true)}>
              Import images
            </Button>
          )}
        </Group>
      </Group>

      {reviewed && (
        <SegmentedControl
          value={split}
          onChange={applySplit}
          data={SPLIT_FILTERS}
          size="xs"
          style={{ alignSelf: 'flex-start' }}
        />
      )}

      {images.isLoading ? (
        <Group justify="center" py="xl">
          <Loader />
        </Group>
      ) : items.length === 0 ? (
        <Card withBorder radius="md" padding="xl">
          <Text size="sm" c="dimmed" ta="center">
            {reviewed
              ? 'Nothing here — review images in the Unreviewed tab, or widen the filter.'
              : 'No unreviewed images. Import some, or widen the filter.'}
          </Text>
        </Card>
      ) : (
        <>
          <ImageGrid
            items={items}
            selected={selected}
            onSelectedChange={setSelected}
            // 지금 보고 있는 목록을 그대로 들고 넘어간다 — 에디터의 위치와
            // 이전·다음이 좁혀 놓은 것을 따라야 한다
            onOpen={(item) =>
              navigate(
                `/projects/${projectId}/datasets/${datasetId}/label/${encodeURIComponent(
                  item.stem,
                )}?${editorSearch(filters, reviewed, split)}`,
              )
            }
          />
          {pages > 1 && (
            <Group justify="center">
              <Pagination value={page} onChange={applyPage} total={pages} />
            </Group>
          )}
          <SelectionBar
            reviewed={reviewed}
            selectedCount={selected.size}
            total={data?.total ?? 0}
            selectingAll={selectAll.isPending}
            reviewPending={sendBack.isPending}
            deletePending={remove.isPending}
            onSelectAll={() => selectAll.mutate()}
            onClear={() => setSelected(new Set())}
            onAutoLabel={() => setAutoLabelOpen(true)}
            onSendBack={() => sendBack.mutate()}
            onDelete={() => {
              if (confirm(`Delete ${selected.size} images from this dataset?`)) remove.mutate()
            }}
          />
        </>
      )}

      <ImportModal
        projectId={projectId}
        datasetId={datasetId}
        opened={importOpen}
        onClose={() => setImportOpen(false)}
      />
      <TilingModal
        projectId={projectId}
        datasetId={datasetId}
        dataset={dataset}
        opened={tileOpen}
        onClose={() => setTileOpen(false)}
      />
      <SplitModal
        projectId={projectId}
        datasetId={datasetId}
        dataset={dataset}
        opened={splitOpen}
        onClose={() => setSplitOpen(false)}
      />
      <ExportModal
        projectId={projectId}
        datasetId={datasetId}
        dataset={dataset}
        opened={exportOpen}
        onClose={() => setExportOpen(false)}
      />
      <AutoLabelModal
        projectId={projectId}
        datasetId={datasetId}
        opened={autoLabelOpen}
        onClose={() => setAutoLabelOpen(false)}
        names={[...selected]}
      />
    </Stack>
  )
}

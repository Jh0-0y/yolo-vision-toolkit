import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ActionIcon,
  Center,
  Group,
  Loader,
  SegmentedControl,
  Select,
  Stack,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import { useDebouncedValue } from '@mantine/hooks'
import { IconArrowsSort, IconSearch, IconSortAscending, IconSortDescending } from '@tabler/icons-react'
import {
  keepPreviousData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { notifications } from '@mantine/notifications'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  api,
  deleteImages,
  listImageNames,
  listImages,
  type ImageQuery,
  type StatsOut,
} from '../api/client'
import StatTile from '../components/StatTile'
import AutoLabelModal from '../components/dataset/AutoLabelModal'
import ExportModal from '../components/dataset/ExportModal'
import ImageGrid from '../components/dataset/ImageGrid'
import SelectionBar from '../components/dataset/SelectionBar'

const PAGE_SIZE = 60

type TriFilter = 'all' | 'yes' | 'no'

function triToBool(v: TriFilter): boolean | undefined {
  return v === 'all' ? undefined : v === 'yes'
}

export default function DatasetPage() {
  const { projectId = '' } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()

  const labeledF = (searchParams.get('labeled') as TriFilter) || 'all'
  const reviewedF = (searchParams.get('reviewed') as TriFilter) || 'all'
  const clsF = searchParams.get('cls') ?? ''
  const sort = (searchParams.get('sort') as 'created' | 'name') || 'created'
  const order = (searchParams.get('order') as 'asc' | 'desc') || 'desc'
  const [search, setSearch] = useState(searchParams.get('q') ?? '')
  const [debouncedSearch] = useDebouncedValue(search, 250)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [autoLabelOpen, setAutoLabelOpen] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)

  const setParam = (key: string, value: string, defaultValue: string) => {
    const next = new URLSearchParams(searchParams)
    if (value === defaultValue) next.delete(key)
    else next.set(key, value)
    setSearchParams(next, { replace: true })
  }

  // keep the search term in the URL so the label editor's prev/next list
  // (and back-navigation) see the same filter set
  useEffect(() => {
    if ((searchParams.get('q') ?? '') !== debouncedSearch) {
      setParam('q', debouncedSearch, '')
    }
  }, [debouncedSearch]) // eslint-disable-line react-hooks/exhaustive-deps

  const filterQuery: ImageQuery = useMemo(
    () => ({
      labeled: triToBool(labeledF),
      reviewed: triToBool(reviewedF),
      cls: clsF === '' ? undefined : Number(clsF),
      sort,
      order,
      q: debouncedSearch || undefined,
    }),
    [labeledF, reviewedF, clsF, sort, order, debouncedSearch],
  )

  const stats = useQuery({
    queryKey: ['stats', projectId],
    queryFn: () => api.get<StatsOut>(`/projects/${projectId}/stats`),
  })

  const images = useInfiniteQuery({
    queryKey: ['images', projectId, filterQuery],
    queryFn: ({ pageParam }) =>
      listImages(projectId, { ...filterQuery, page: pageParam, size: PAGE_SIZE }),
    initialPageParam: 1,
    getNextPageParam: (last) =>
      last.page * last.size < last.total ? last.page + 1 : undefined,
    // keep showing the previous grid while a filter change refetches —
    // prevents the whole page collapsing and re-expanding on every toggle
    placeholderData: keepPreviousData,
  })

  const items = useMemo(
    () => images.data?.pages.flatMap((p) => p.items) ?? [],
    [images.data],
  )
  const total = images.data?.pages[0]?.total ?? 0

  // infinite scroll sentinel
  const sentinelRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const io = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && images.hasNextPage && !images.isFetchingNextPage) {
        images.fetchNextPage()
      }
    })
    io.observe(el)
    return () => io.disconnect()
  }, [images.hasNextPage, images.isFetchingNextPage, images.fetchNextPage]) // eslint-disable-line react-hooks/exhaustive-deps

  const selectAll = async () => {
    const { names } = await listImageNames(projectId, filterQuery)
    setSelected(new Set(names))
  }

  const remove = useMutation({
    mutationFn: (names: string[]) => deleteImages(projectId, names),
    onSuccess: (res) => {
      notifications.show({ message: `${res.deleted} images deleted`, color: 'green' })
      setSelected(new Set())
      queryClient.invalidateQueries({ queryKey: ['images', projectId] })
      queryClient.invalidateQueries({ queryKey: ['image-names', projectId] })
      queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const selectedNames = selected.size > 0 ? [...selected] : null

  return (
    <Stack gap="md" pb={80}>
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={3}>Dataset</Title>
          <Group gap="sm" mt="xs">
            <StatTile label="Images" value={stats.data?.images ?? '–'} color="gray.7" />
            <StatTile label="Labeled" value={stats.data?.labeled ?? '–'} color="blue" />
            <StatTile label="Reviewed" value={stats.data?.reviewed ?? '–'} color="teal" />
            <StatTile
              label="Classes"
              value={stats.data?.classes.length ?? '–'}
              color="grape"
              title={
                stats.data && stats.data.classes.length > 0
                  ? stats.data.classes.map((c) => `${c.id}: ${c.name}`).join(', ')
                  : undefined
              }
            />
          </Group>
        </div>
      </Group>

      <Group gap="sm" wrap="wrap">
        <TextInput
          size="xs"
          placeholder="Search filename"
          leftSection={<IconSearch size={14} />}
          value={search}
          onChange={(e) => setSearch(e.currentTarget.value)}
          w={180}
        />
        <Group gap={6}>
          <Text size="xs" c="dimmed">
            Labeled
          </Text>
          <SegmentedControl
            size="xs"
            value={labeledF}
            onChange={(v) => setParam('labeled', v, 'all')}
            data={[
              { value: 'all', label: 'All' },
              { value: 'yes', label: 'Yes' },
              { value: 'no', label: 'No' },
            ]}
          />
        </Group>
        <Group gap={6}>
          <Text size="xs" c="dimmed">
            Reviewed
          </Text>
          <SegmentedControl
            size="xs"
            value={reviewedF}
            onChange={(v) => setParam('reviewed', v, 'all')}
            data={[
              { value: 'all', label: 'All' },
              { value: 'yes', label: 'Yes' },
              { value: 'no', label: 'No' },
            ]}
          />
        </Group>
        <Select
          size="xs"
          w={150}
          placeholder="All classes"
          value={clsF || null}
          onChange={(v) => setParam('cls', v ?? '', '')}
          data={[
            { value: '-1', label: 'No class (negative)' },
            ...(stats.data?.classes.map((c) => ({
              value: String(c.id),
              label: `${c.id}: ${c.name}`,
            })) ?? []),
          ]}
          clearable
        />
        <Group gap={4}>
          <Select
            size="xs"
            w={120}
            leftSection={<IconArrowsSort size={14} />}
            value={sort}
            onChange={(v) => setParam('sort', v ?? 'created', 'created')}
            data={[
              { value: 'created', label: 'Created' },
              { value: 'name', label: 'Name' },
            ]}
            allowDeselect={false}
          />
          <Tooltip label={order === 'desc' ? 'Descending' : 'Ascending'}>
            <ActionIcon
              variant="default"
              size="input-xs"
              onClick={() => setParam('order', order === 'desc' ? 'asc' : 'desc', 'desc')}
            >
              {order === 'desc' ? <IconSortDescending size={16} /> : <IconSortAscending size={16} />}
            </ActionIcon>
          </Tooltip>
        </Group>
        <Text size="xs" c="dimmed" ml="auto">
          {total} images
        </Text>
      </Group>

      {images.isSuccess && items.length === 0 && (
        <Text c="dimmed" py="lg">
          No images match the filters. Add images in Upload Data.
        </Text>
      )}

      {items.length > 0 && (
        <ImageGrid
          items={items}
          selected={selected}
          onSelectedChange={setSelected}
          onOpen={(img) =>
            navigate(`label/${encodeURIComponent(img.stem)}?${searchParams.toString()}`)
          }
        />
      )}

      <div ref={sentinelRef} />
      {images.isFetchingNextPage && (
        <Center py="sm">
          <Loader size="sm" />
        </Center>
      )}

      <SelectionBar
        selectedCount={selected.size}
        total={total}
        onSelectAll={selectAll}
        onClear={() => setSelected(new Set())}
        onAutoLabel={() => setAutoLabelOpen(true)}
        onExport={() => setExportOpen(true)}
        onDelete={() => {
          if (selected.size === 0) return
          if (confirm(`Delete ${selected.size} selected images and their labels? This cannot be undone.`))
            remove.mutate([...selected])
        }}
      />

      <AutoLabelModal
        projectId={projectId}
        opened={autoLabelOpen}
        onClose={() => setAutoLabelOpen(false)}
        names={selectedNames}
      />
      {exportOpen && (
        <ExportModal
          projectId={projectId}
          opened
          onClose={() => setExportOpen(false)}
          names={selectedNames}
        />
      )}
    </Stack>
  )
}

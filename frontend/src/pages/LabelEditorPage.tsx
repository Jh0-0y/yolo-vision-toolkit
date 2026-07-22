import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  ScrollArea,
  SegmentedControl,
  Select,
  Stack,
  Switch,
  Text,
  Tooltip,
} from '@mantine/core'
import {
  IconArrowLeft,
  IconChevronLeft,
  IconChevronRight,
  IconPointer,
  IconSquarePlus,
  IconTrash,
} from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  getLabels,
  listImageNames,
  putLabels,
  putReviewed,
  type ImageQuery,
} from '../api/client'
import BBoxCanvas from '../components/editor/BBoxCanvas'
import { classColor, useEditorStore, type EditorTool } from '../stores/editorStore'

type TriFilter = 'all' | 'yes' | 'no'

function triToBool(v: string | null): boolean | undefined {
  return v === 'yes' ? true : v === 'no' ? false : undefined
}

export default function LabelEditorPage() {
  const { projectId = '', stem = '' } = useParams()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const store = useEditorStore()
  const { boxes, selectedId, tool, dirty } = store

  const canvasBox = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 800, h: 560 })
  const [reviewed, setReviewed] = useState(false)

  // the filtered/sorted name list this editor navigates within (frozen per filter set)
  const filterQuery: ImageQuery = useMemo(
    () => ({
      labeled: triToBool(searchParams.get('labeled') as TriFilter),
      reviewed: triToBool(searchParams.get('reviewed') as TriFilter),
      cls: searchParams.get('cls') != null ? Number(searchParams.get('cls')) : undefined,
      sort: (searchParams.get('sort') as 'created' | 'name') || 'created',
      order: (searchParams.get('order') as 'asc' | 'desc') || 'desc',
      q: searchParams.get('q') || undefined,
    }),
    [searchParams],
  )
  const names = useQuery({
    queryKey: ['image-names', projectId, filterQuery],
    queryFn: () => listImageNames(projectId, filterQuery),
    staleTime: 5 * 60_000,
  })

  const nameList = names.data?.names ?? []
  const index = nameList.findIndex((n) => n.replace(/\.[^.]+$/, '') === stem)
  const prevStem = index > 0 ? nameList[index - 1].replace(/\.[^.]+$/, '') : null
  const nextStem =
    index >= 0 && index < nameList.length - 1
      ? nameList[index + 1].replace(/\.[^.]+$/, '')
      : null

  const detail = useQuery({
    queryKey: ['labels', projectId, stem],
    queryFn: () => getLabels(projectId, stem),
  })

  // load boxes into the shared editor store when the image arrives
  const loadedFor = useRef<string | null>(null)
  useEffect(() => {
    const d = detail.data
    if (d && loadedFor.current !== d.stem) {
      loadedFor.current = d.stem
      setReviewed(d.reviewed)
      store.load(d.boxes.map((b, i) => ({ ...b, id: b.id ?? `b${i}` })) as never)
    }
  }, [detail.data]) // eslint-disable-line react-hooks/exhaustive-deps

  // measure canvas container
  useLayoutEffect(() => {
    const el = canvasBox.current
    if (!el) return
    const update = () =>
      setSize({
        w: el.clientWidth,
        h: Math.max(420, Math.round(window.innerHeight - 280)),
      })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [])

  // ---------- saving ----------

  const saveNow = useCallback(async () => {
    const s = useEditorStore.getState()
    if (!s.dirty) return
    await putLabels(projectId, stem, s.boxes as never)
    s.markSaved()
    queryClient.invalidateQueries({ queryKey: ['images', projectId] })
    queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    queryClient.invalidateQueries({ queryKey: ['labels', projectId, stem] })
  }, [projectId, stem, queryClient])

  // 2s debounce autosave while editing
  useEffect(() => {
    if (!dirty) return
    const t = setTimeout(() => {
      saveNow().catch((e) => notifications.show({ message: String(e), color: 'red' }))
    }, 2000)
    return () => clearTimeout(t)
  }, [boxes, dirty, saveNow])

  const goTo = useCallback(
    (targetStem: string | null) => {
      if (!targetStem) return
      saveNow().catch((e) => notifications.show({ message: String(e), color: 'red' }))
      navigate(
        `/projects/${projectId}/dataset/label/${encodeURIComponent(targetStem)}?${searchParams.toString()}`,
        { replace: true },
      )
    },
    [navigate, projectId, searchParams, saveNow],
  )

  const backToDataset = () => {
    saveNow().catch((e) => notifications.show({ message: String(e), color: 'red' }))
    navigate(`/projects/${projectId}/dataset?${searchParams.toString()}`)
  }

  const toggleReviewed = useMutation({
    mutationFn: async (flag: boolean) => {
      await saveNow()
      return putReviewed(projectId, stem, flag)
    },
    onSuccess: (res) => {
      setReviewed(res.reviewed)
      queryClient.invalidateQueries({ queryKey: ['images', projectId] })
      queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  // ---------- keyboard ----------

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) return
      const s = useEditorStore.getState()
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) s.redo()
        else s.undo()
        return
      }
      if (e.key === 'ArrowLeft') goTo(prevStem)
      else if (e.key === 'ArrowRight') goTo(nextStem)
      else if (e.key === 'v' || e.key === 'V') s.setTool('select')
      else if (e.key === 'w' || e.key === 'W') s.setTool(s.tool === 'draw' ? 'select' : 'draw')
      else if (e.key === 'r' || e.key === 'R') toggleReviewed.mutate(!reviewed)
      else if (['d', 'D', 'Delete', 'Backspace'].includes(e.key) && s.selectedId)
        s.deleteBox(s.selectedId)
      else if (e.key === 'Escape') {
        s.setTool('select')
        s.select(null)
      } else if (/^[1-9]$/.test(e.key) && detail.data) {
        const cls = detail.data.classes[Number(e.key) - 1]
        if (cls) {
          if (s.selectedId) s.updateBox(s.selectedId, { cls: cls.id })
          else s.setActiveCls(cls.id)
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [prevStem, nextStem, goTo, detail.data, reviewed]) // eslint-disable-line react-hooks/exhaustive-deps

  const d = detail.data
  const classOptions =
    d?.classes.map((c) => ({ value: String(c.id), label: `${c.id}: ${c.name}` })) ?? []
  const selectedBox = boxes.find((b) => b.id === selectedId)

  return (
    <Stack gap="xs">
      <Group justify="space-between" wrap="nowrap">
        <Group gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
          <Tooltip label="Back to Dataset">
            <ActionIcon variant="default" onClick={backToDataset}>
              <IconArrowLeft size={16} />
            </ActionIcon>
          </Tooltip>
          <Text fw={600} truncate>
            {d?.name ?? stem}
          </Text>
          <Badge variant="light" color={dirty ? 'yellow' : 'gray'} miw={64} ta="center">
            {dirty ? 'Saving' : 'Saved'}
          </Badge>
        </Group>

        <Group gap="xs" wrap="nowrap">
          <Group gap={4} wrap="nowrap">
            <Tooltip label="Previous (←)">
              <ActionIcon variant="default" disabled={!prevStem} onClick={() => goTo(prevStem)}>
                <IconChevronLeft size={16} />
              </ActionIcon>
            </Tooltip>
            <Text size="xs" c="dimmed" w={70} ta="center">
              {index >= 0 ? `${index + 1} / ${nameList.length}` : '–'}
            </Text>
            <Tooltip label="Next (→)">
              <ActionIcon variant="default" disabled={!nextStem} onClick={() => goTo(nextStem)}>
                <IconChevronRight size={16} />
              </ActionIcon>
            </Tooltip>
          </Group>
          <Switch
            label="Reviewed (R)"
            checked={reviewed}
            onChange={(e) => toggleReviewed.mutate(e.currentTarget.checked)}
            disabled={toggleReviewed.isPending}
          />
        </Group>
      </Group>

      <Group gap="xs">
        <SegmentedControl
          size="xs"
          value={tool}
          onChange={(v) => store.setTool(v as EditorTool)}
          data={[
            {
              value: 'select',
              label: (
                <Group gap={4} wrap="nowrap">
                  <IconPointer size={14} /> <span>Select (V)</span>
                </Group>
              ),
            },
            {
              value: 'draw',
              label: (
                <Group gap={4} wrap="nowrap">
                  <IconSquarePlus size={14} /> <span>Draw (W)</span>
                </Group>
              ),
            },
          ]}
        />
        <Select
          size="xs"
          w={180}
          searchable
          placeholder="Class"
          data={classOptions}
          value={selectedBox ? String(selectedBox.cls) : String(store.activeCls)}
          onChange={(v) => {
            if (v == null) return
            if (selectedBox) store.updateBox(selectedBox.id, { cls: Number(v) })
            else store.setActiveCls(Number(v))
          }}
        />
        {selectedBox && (
          <Button
            size="xs"
            color="red"
            variant="light"
            leftSection={<IconTrash size={14} />}
            onClick={() => store.deleteBox(selectedBox.id)}
          >
            Delete (D)
          </Button>
        )}
        <Text size="xs" c="dimmed" ml="auto">
          Wheel zoom · drag pan · <b>V/W</b> tools · <b>D</b> delete · <b>1-9</b> class ·{' '}
          <b>R</b> reviewed · <b>⌘Z</b> undo
        </Text>
      </Group>

      <Group align="flex-start" gap="sm" wrap="nowrap">
        <div
          ref={canvasBox}
          style={{ flex: 1, minWidth: 0, borderRadius: 8, overflow: 'hidden' }}
        >
          {d?.image_url && <BBoxCanvas imageUrl={d.image_url} width={size.w} height={size.h} />}
        </div>

        <Card withBorder radius="md" padding="sm" w={240} style={{ flexShrink: 0 }}>
          <Text size="sm" fw={600} mb="xs">
            {boxes.length} boxes
          </Text>
          <ScrollArea.Autosize mah={size.h - 60}>
            <Stack gap={4}>
              {boxes.map((b) => {
                const cls = d?.classes.find((c) => c.id === b.cls)
                return (
                  <Group
                    key={b.id}
                    gap="xs"
                    wrap="nowrap"
                    onClick={() => {
                      store.setTool('select')
                      store.select(b.id)
                    }}
                    style={{
                      cursor: 'pointer',
                      borderRadius: 6,
                      padding: '4px 6px',
                      background:
                        b.id === selectedId ? 'var(--mantine-color-blue-light)' : undefined,
                    }}
                  >
                    <div
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 3,
                        background: classColor(b.cls),
                        flexShrink: 0,
                      }}
                    />
                    <Text size="xs" truncate style={{ flex: 1 }}>
                      {cls ? cls.name : `class ${b.cls}`}
                    </Text>
                    {b.score != null && (
                      <Text size="xs" c="dimmed">
                        {(b.score * 100).toFixed(0)}%
                      </Text>
                    )}
                  </Group>
                )
              })}
              {boxes.length === 0 && (
                <Text size="xs" c="dimmed">
                  No boxes yet. Use the draw tool (W) to add one.
                </Text>
              )}
            </Stack>
          </ScrollArea.Autosize>
        </Card>
      </Group>
    </Stack>
  )
}

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  Modal,
  ScrollArea,
  SegmentedControl,
  Select,
  Stack,
  Switch,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core'
import {
  IconArrowLeft,
  IconChevronLeft,
  IconChevronRight,
  IconPlus,
  IconPointer,
  IconSquarePlus,
  IconTrash,
} from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  addDatasetClass,
  deleteDatasetImages,
  getDatasetLabels,
  listDatasetImages,
  putDatasetLabels,
  setImageReviewed,
  type DatasetImageQuery,
  type DatasetLabels,
} from '../api/client'
import BBoxCanvas from '../components/editor/BBoxCanvas'
import { classColor, useEditorStore, type EditorTool } from '../stores/editorStore'

type TriFilter = 'all' | 'yes' | 'no'

function triToBool(v: string | null): boolean | undefined {
  return v === 'yes' ? true : v === 'no' ? false : undefined
}

export default function LabelEditorPage() {
  const { projectId = '', datasetId = '', stem = '' } = useParams()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const store = useEditorStore()
  const { boxes, selectedId, tool, dirty } = store

  const canvasBox = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 800, h: 560 })
  const [addClassOpen, setAddClassOpen] = useState(false)
  const [newClassName, setNewClassName] = useState('')

  // 이 에디터가 오가는 목록 — **그리드에서 좁혀 놓은 그대로**여야 한다.
  // 조건이 하나라도 빠지면 `3 / 128` 의 총 수가 달라지고 이전·다음이 엉뚱한 데로 간다.
  const filterQuery: DatasetImageQuery = useMemo(() => {
    const sort = searchParams.get('sort')
    const order = searchParams.get('order')
    const split = searchParams.get('split')
    return {
      labeled: triToBool(searchParams.get('labeled') as TriFilter),
      reviewed: triToBool(searchParams.get('reviewed') as TriFilter),
      cls: searchParams.get('cls') != null ? Number(searchParams.get('cls')) : undefined,
      q: searchParams.get('q') || undefined,
      sort: sort === 'name' ? 'name' : 'created',
      order: order === 'asc' ? 'asc' : 'desc',
      ...(split ? { split: split as DatasetImageQuery['split'] } : {}),
      names_only: true,
    }
  }, [searchParams])
  const names = useQuery({
    queryKey: ['image-names', projectId, datasetId, filterQuery],
    queryFn: () => listDatasetImages(projectId, datasetId, filterQuery),
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
    queryKey: ['labels', projectId, datasetId, stem],
    queryFn: () => getDatasetLabels(projectId, datasetId, stem),
  })

  // 검수 여부는 **캐시가 진실이다.** 따로 useState 로 복사해 두면 이전·다음으로
  // 오갈 때 캐시에 남은 옛 값이 스위치를 되돌려 놓는다.
  const reviewed = detail.data?.reviewed ?? false

  // load boxes into the shared editor store when the image arrives
  const loadedFor = useRef<string | null>(null)
  useEffect(() => {
    const d = detail.data
    if (d && loadedFor.current !== d.stem) {
      loadedFor.current = d.stem
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
    await putDatasetLabels(projectId, datasetId, stem, s.boxes as never)
    s.markSaved()
    // 박스가 바뀌면 그리드의 배지(L·N)와 클래스 필터 결과가 달라진다
    queryClient.invalidateQueries({ queryKey: ['dataset-images', projectId, datasetId] })
    queryClient.invalidateQueries({ queryKey: ['labels', projectId, datasetId, stem] })
  }, [projectId, datasetId, stem, queryClient])

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
        `/projects/${projectId}/datasets/${datasetId}/label/${encodeURIComponent(targetStem)}?${searchParams.toString()}`,
        { replace: true },
      )
    },
    [navigate, projectId, datasetId, searchParams, saveNow],
  )

  const backToDataset = () => {
    saveNow().catch((e) => notifications.show({ message: String(e), color: 'red' }))
    // 좁혀 놓은 목록으로 돌아간다 — 필터를 버리면 어디를 보고 있었는지 잃는다.
    // 검수 여부는 그리드에서 `tab` 이라 이름이 다르다.
    const back = new URLSearchParams(searchParams)
    back.set('tab', searchParams.get('reviewed') === 'yes' ? 'reviewed' : 'unreviewed')
    back.delete('reviewed')
    navigate(`/projects/${projectId}/datasets/${datasetId}?${back.toString()}`)
  }

  const toggleReviewed = useMutation({
    mutationFn: async (flag: boolean) => {
      await saveNow()
      return setImageReviewed(projectId, datasetId, stem, flag)
    },
    onSuccess: (res) => {
      // 이 이미지의 상세 캐시를 바로 고쳐 둔다 — 다시 돌아왔을 때 옛 값을 읽지 않도록.
      queryClient.setQueryData<DatasetLabels>(
        ['labels', projectId, datasetId, stem],
        (old) => (old ? { ...old, reviewed: res.reviewed } : old),
      )
      // 검수를 켜면 이 이미지는 검수완료 탭으로 넘어간다 — 수치도 함께 다시 읽는다
      queryClient.invalidateQueries({ queryKey: ['dataset-images', projectId, datasetId] })
      queryClient.invalidateQueries({ queryKey: ['dataset', projectId, datasetId] })
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const removeImage = useMutation({
    mutationFn: () => deleteDatasetImages(projectId, datasetId, [detail.data?.name ?? '']),
    onSuccess: () => {
      // the deleted stem's autosave must not recreate its label file
      useEditorStore.getState().markSaved()
      notifications.show({ message: `Deleted ${detail.data?.name ?? stem}`, color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['dataset-images', projectId, datasetId] })
      queryClient.invalidateQueries({ queryKey: ['image-names', projectId, datasetId] })
      queryClient.invalidateQueries({ queryKey: ['dataset', projectId, datasetId] })
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
      queryClient.removeQueries({ queryKey: ['labels', projectId, datasetId, stem] })
      // jump to a neighbour to keep the flow going, else back to the gallery
      const target = nextStem ?? prevStem
      if (target) {
        navigate(
          `/projects/${projectId}/datasets/${datasetId}/label/${encodeURIComponent(target)}?${searchParams.toString()}`,
          { replace: true },
        )
      } else {
        backToDataset()
      }
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const createClass = useMutation({
    mutationFn: (name: string) => addDatasetClass(projectId, datasetId, name),
    onSuccess: (cls) => {
      notifications.show({ message: `Class added: ${cls.name}`, color: 'green' })
      // refresh the label detail (its classes) and the active class becomes the new one
      queryClient.invalidateQueries({ queryKey: ['labels', projectId, datasetId, stem] })
      queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
      const s = useEditorStore.getState()
      if (s.selectedId) s.updateBox(s.selectedId, { cls: cls.id })
      else s.setActiveCls(cls.id)
      setAddClassOpen(false)
      setNewClassName('')
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  // ---------- keyboard ----------

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) return
      const s = useEditorStore.getState()
      // Match by physical key (e.code) so shortcuts work regardless of the active
      // input source — e.key is remapped by the Korean IME (w -> ㅈ, r -> ㄱ ...).
      const mod = e.metaKey || e.ctrlKey

      if (mod) {
        if (e.code === 'KeyZ') {
          e.preventDefault()
          if (e.shiftKey) s.redo()
          else s.undo()
        } else if (e.code === 'KeyC' && s.selectedId) {
          e.preventDefault()
          s.copy()
        } else if (e.code === 'KeyV' && s.clipboard.length) {
          e.preventDefault()
          s.paste()
        }
        return // let the browser own every other cmd/ctrl combo
      }

      if (e.code === 'ArrowLeft') goTo(prevStem)
      else if (e.code === 'ArrowRight') goTo(nextStem)
      else if (e.code === 'KeyV') s.setTool('select')
      else if (e.code === 'KeyW') s.setTool(s.tool === 'draw' ? 'select' : 'draw')
      else if (e.code === 'KeyR') toggleReviewed.mutate(!reviewed)
      else if ((e.code === 'KeyD' || e.code === 'Delete' || e.code === 'Backspace') && s.selectedId)
        s.deleteBox(s.selectedId)
      else if (e.code === 'Escape') {
        s.setTool('select')
        s.select(null)
      } else if (/^Digit[1-9]$/.test(e.code) && detail.data) {
        const cls = detail.data.classes[Number(e.code.slice(5)) - 1]
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
          <Tooltip label="Delete this image (and its labels)">
            <ActionIcon
              color="red"
              variant="light"
              disabled={!detail.data || removeImage.isPending}
              loading={removeImage.isPending}
              onClick={() => {
                const name = detail.data?.name
                if (!name) return
                if (confirm(`Delete "${name}"?\nThe image and its labels will be removed.`))
                  removeImage.mutate()
              }}
            >
              <IconTrash size={16} />
            </ActionIcon>
          </Tooltip>
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
          placeholder="Class"
          data={classOptions}
          value={selectedBox ? String(selectedBox.cls) : String(store.activeCls)}
          onChange={(v) => {
            if (v == null) return
            if (selectedBox) store.updateBox(selectedBox.id, { cls: Number(v) })
            else store.setActiveCls(Number(v))
          }}
        />
        <Tooltip label="Add a new class">
          <Button
            size="xs"
            variant="light"
            leftSection={<IconPlus size={14} />}
            onClick={() => setAddClassOpen(true)}
          >
            Class
          </Button>
        </Tooltip>
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
          <b>R</b> reviewed · <b>⌘C/⌘V</b> copy/paste · <b>⌘Z</b> undo
        </Text>
      </Group>

      <Group align="flex-start" gap="sm" wrap="nowrap">
        <div
          ref={canvasBox}
          style={{ flex: 1, minWidth: 0, borderRadius: 8, overflow: 'hidden' }}
        >
          {d?.image_url && (
            <BBoxCanvas
              imageUrl={d.image_url}
              width={size.w}
              height={size.h}
              classNames={Object.fromEntries((d?.classes ?? []).map((c) => [c.id, c.name]))}
            />
          )}
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

      <Modal
        opened={addClassOpen}
        onClose={() => setAddClassOpen(false)}
        title="New class"
        size="sm"
      >
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const name = newClassName.trim()
            if (name && !createClass.isPending) createClass.mutate(name)
          }}
        >
          <Stack>
            <TextInput
              label="Class name"
              placeholder="e.g. player"
              value={newClassName}
              onChange={(e) => setNewClassName(e.currentTarget.value)}
              data-autofocus
            />
            <Button type="submit" disabled={!newClassName.trim()} loading={createClass.isPending}>
              Add
            </Button>
          </Stack>
        </form>
      </Modal>
    </Stack>
  )
}

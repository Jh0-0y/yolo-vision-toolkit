import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Badge, Button, Group, Modal, Select, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getLabels, putLabels, type LabelDetail } from '../../api/client'
import { useEditorStore } from '../../stores/editorStore'
import BBoxCanvas from './BBoxCanvas'

interface Props {
  projectId: string
  bucket: 'confirmed' | 'review'
  stem: string | null
  onClose: () => void
}

export default function LabelEditModal({ projectId, bucket, stem, onClose }: Props) {
  const queryClient = useQueryClient()
  const store = useEditorStore()
  const { boxes, selectedId, drawMode, dirty } = store
  const canvasBox = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 800, h: 560 })
  const [saving, setSaving] = useState(false)

  const opened = stem != null
  // Latch the stem being edited so the modal keeps rendering its image/boxes
  // through Mantine's close animation. If we cleared content the instant stem
  // went null, the canvas would collapse mid-transition and the modal would
  // freeze half-rendered.
  const [activeStem, setActiveStem] = useState<string | null>(stem)
  useEffect(() => {
    if (stem != null) setActiveStem(stem)
  }, [stem])

  const detail = useQuery({
    queryKey: ['labels', projectId, bucket, activeStem],
    queryFn: () => getLabels(projectId, bucket, activeStem as string),
    enabled: activeStem != null,
  })

  // load boxes into the shared editor store when the image arrives
  const loadedFor = useRef<string | null>(null)
  useEffect(() => {
    const d = detail.data
    if (opened && d && loadedFor.current !== d.stem) {
      loadedFor.current = d.stem
      store.load(d.boxes.map((b, i) => ({ ...b, id: b.id ?? `b${i}` })) as never)
    }
    if (!opened) loadedFor.current = null
  }, [detail.data, opened]) // eslint-disable-line react-hooks/exhaustive-deps

  // measure canvas container once the modal body is mounted
  useLayoutEffect(() => {
    if (!opened) return
    const el = canvasBox.current
    if (!el) return
    const update = () =>
      setSize({ w: el.clientWidth, h: Math.max(360, Math.round(window.innerHeight * 0.6)) })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [opened])

  // modal-scoped keyboard shortcuts
  useEffect(() => {
    if (!opened) return
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) return
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        e.shiftKey ? store.redo() : store.undo()
        return
      }
      if (e.key === 'w' || e.key === 'W') store.setDrawMode(!store.drawMode)
      else if (['d', 'D', 'Delete', 'Backspace'].includes(e.key) && store.selectedId)
        store.deleteBox(store.selectedId)
      else if (e.key === 'Escape') {
        store.setDrawMode(false)
        store.select(null)
      } else if (/^[1-9]$/.test(e.key) && store.selectedId && detail.data) {
        const cls = detail.data.classes[Number(e.key) - 1]
        if (cls) store.updateBox(store.selectedId, { cls: cls.id })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [opened, detail.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const save = async () => {
    if (!stem) return
    setSaving(true)
    try {
      await putLabels(projectId, bucket, stem, boxes as never)
      store.markSaved()
      notifications.show({ message: '라벨 저장됨', color: 'green' })
      queryClient.invalidateQueries({ queryKey: ['images', projectId] })
      queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
      onClose()
    } catch (e) {
      notifications.show({ message: String(e), color: 'red' })
    } finally {
      setSaving(false)
    }
  }

  const d: LabelDetail | undefined = detail.data
  const classOptions =
    d?.classes.map((c) => ({ value: String(c.id), label: `${c.id}: ${c.name}` })) ?? []
  const selectedBox = boxes.find((b) => b.id === selectedId)

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      size="90%"
      title={
        <Group gap="xs">
          <Text fw={600}>라벨 편집 · {activeStem}</Text>
          <Badge variant="light" color={bucket === 'confirmed' ? 'green' : 'orange'}>
            {bucket === 'confirmed' ? '확정' : '검수 대기'}
          </Badge>
          {dirty && (
            <Badge variant="light" color="yellow">
              변경됨
            </Badge>
          )}
        </Group>
      }
    >
      <Stack gap="xs">
        <Group justify="space-between">
          <Group gap="xs">
            <Button
              size="xs"
              variant={drawMode ? 'filled' : 'light'}
              onClick={() => store.setDrawMode(!drawMode)}
            >
              박스 그리기 (W)
            </Button>
            <Select
              size="xs"
              w={180}
              searchable
              placeholder="클래스"
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
                onClick={() => store.deleteBox(selectedBox.id)}
              >
                삭제 (D)
              </Button>
            )}
          </Group>
          <Group gap="xs">
            <Text size="xs" c="dimmed">
              박스 {boxes.length}개
            </Text>
            <Button size="xs" variant="default" onClick={onClose}>
              닫기
            </Button>
            <Button size="xs" onClick={save} loading={saving}>
              저장
            </Button>
          </Group>
        </Group>

        <div ref={canvasBox} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }}>
          {d?.image_url && <BBoxCanvas imageUrl={d.image_url} width={size.w} height={size.h} />}
        </div>

        <Text size="xs" c="dimmed">
          휠 줌 · 드래그 팬(이미지 범위 내) · <b>W</b> 박스 그리기 · <b>D</b> 삭제 · <b>1-9</b>{' '}
          클래스 · <b>⌘Z</b> 되돌리기
        </Text>
      </Stack>
    </Modal>
  )
}

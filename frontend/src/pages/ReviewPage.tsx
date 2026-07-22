import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  Group,
  Image,
  Kbd,
  ScrollArea,
  Select,
  Stack,
  Text,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import {
  api,
  type ReviewItemDetail,
  type ReviewQueue,
} from '../api/client'
import BBoxCanvas from '../components/editor/BBoxCanvas'
import { classColor, useEditorStore } from '../stores/editorStore'

export default function ReviewPage() {
  const { projectId = '' } = useParams()
  const queryClient = useQueryClient()
  const [currentId, setCurrentId] = useState<string | null>(null)
  const canvasBox = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 600 })

  const store = useEditorStore()
  const { boxes, selectedId, drawMode, dirty } = store

  const queue = useQuery({
    queryKey: ['review-queue', projectId],
    queryFn: () => api.get<ReviewQueue>(`/projects/${projectId}/review?size=500`),
  })

  // auto-select the first queue item
  useEffect(() => {
    if (!currentId && queue.data?.items.length) setCurrentId(queue.data.items[0].id)
  }, [queue.data, currentId])

  const item = useQuery({
    queryKey: ['review-item', currentId],
    queryFn: () => api.get<ReviewItemDetail>(`/review/${currentId}`),
    enabled: !!currentId,
  })

  // load boxes into the editor when a new item arrives
  const loadedFor = useRef<string | null>(null)
  useEffect(() => {
    if (item.data && loadedFor.current !== item.data.id) {
      loadedFor.current = item.data.id
      store.load(
        item.data.boxes.map((b, i) => ({ ...b, id: b.id ?? `b${i}` })) as never,
      )
    }
  }, [item.data]) // eslint-disable-line react-hooks/exhaustive-deps

  // measure canvas container
  useLayoutEffect(() => {
    const el = canvasBox.current
    if (!el) return
    const update = () =>
      setCanvasSize({ w: el.clientWidth, h: Math.max(400, window.innerHeight - 180) })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // debounced draft autosave
  useEffect(() => {
    if (!dirty || !currentId) return
    const t = setTimeout(() => {
      api.put(`/review/${currentId}`, { boxes }).then(() => store.markSaved())
    }, 2000)
    return () => clearTimeout(t)
  }, [boxes, dirty, currentId]) // eslint-disable-line react-hooks/exhaustive-deps

  const advance = (nextId: string | null) => {
    loadedFor.current = null
    queryClient.invalidateQueries({ queryKey: ['review-queue', projectId] })
    queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    setCurrentId(nextId)
  }

  const approve = useMutation({
    mutationFn: () =>
      api.post<{ next_id: string | null }>(`/review/${currentId}/approve`, { boxes }),
    onSuccess: (r) => {
      notifications.show({ message: '승인됨 → 확정 데이터로 이동', color: 'green' })
      advance(r.next_id)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const reject = useMutation({
    mutationFn: () => api.post<{ next_id: string | null }>(`/review/${currentId}/reject`),
    onSuccess: (r) => {
      notifications.show({ message: '반려됨', color: 'yellow' })
      advance(r.next_id)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const navQueue = useCallback(
    (dir: 1 | -1) => {
      const items = queue.data?.items ?? []
      const idx = items.findIndex((i) => i.id === currentId)
      const next = items[idx + dir]
      if (next) {
        loadedFor.current = null
        setCurrentId(next.id)
      }
    },
    [queue.data, currentId],
  )

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
        return
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        e.shiftKey ? store.redo() : store.undo()
        return
      }
      switch (e.key) {
        case 'a':
        case 'A':
          if (currentId && !approve.isPending) approve.mutate()
          break
        case 'x':
        case 'X':
          if (currentId && !reject.isPending) reject.mutate()
          break
        case 'd':
        case 'D':
        case 'Delete':
        case 'Backspace':
          if (selectedId) store.deleteBox(selectedId)
          break
        case 'w':
        case 'W':
          store.setDrawMode(!drawMode)
          break
        case 'Escape':
          store.setDrawMode(false)
          store.select(null)
          break
        case 'ArrowRight':
          navQueue(1)
          break
        case 'ArrowLeft':
          navQueue(-1)
          break
        default:
          if (/^[1-9]$/.test(e.key) && selectedId && item.data) {
            const cls = item.data.classes[Number(e.key) - 1]
            if (cls) store.updateBox(selectedId, { cls: cls.id })
          }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [currentId, selectedId, drawMode, item.data, navQueue]) // eslint-disable-line react-hooks/exhaustive-deps

  const selectedBox = boxes.find((b) => b.id === selectedId)
  const classOptions =
    item.data?.classes.map((c) => ({ value: String(c.id), label: `${c.id}: ${c.name}` })) ?? []

  if (queue.data && queue.data.total === 0) {
    return (
      <Stack align="center" py="xl">
        <Text size="lg">리뷰할 항목이 없습니다 🎉</Text>
        <Button component={Link} to={`/projects/${projectId}`} variant="light">
          프로젝트로 돌아가기
        </Button>
      </Stack>
    )
  }

  return (
    <Group align="stretch" gap="xs" wrap="nowrap" h="100%">
      {/* left: queue */}
      <Card withBorder w={210} p="xs">
        <Group justify="space-between" mb="xs">
          <Text fw={600} size="sm">
            리뷰 큐
          </Text>
          <Badge variant="light">{queue.data?.total ?? 0}</Badge>
        </Group>
        <ScrollArea h="calc(100vh - 200px)">
          <Stack gap={6}>
            {queue.data?.items.map((q) => (
              <Card
                key={q.id}
                p={4}
                withBorder
                style={{
                  cursor: 'pointer',
                  borderColor: q.id === currentId ? 'var(--mantine-color-blue-5)' : undefined,
                }}
                onClick={() => {
                  loadedFor.current = null
                  setCurrentId(q.id)
                }}
              >
                {q.thumb && <Image src={q.thumb} radius="xs" h={80} fit="cover" />}
                <Text size="xs" truncate>
                  {q.stem}
                </Text>
                <Text size="xs" c="dimmed">
                  불확실도 {(q.uncertainty * 100).toFixed(0)}% · 플래그 {q.n_flagged}
                </Text>
              </Card>
            ))}
          </Stack>
        </ScrollArea>
      </Card>

      {/* center: canvas */}
      <Stack style={{ flex: 1, minWidth: 0 }} gap="xs">
        <Group justify="space-between">
          <Group gap="xs">
            <Text fw={600}>{item.data?.stem}</Text>
            {dirty && (
              <Badge color="yellow" variant="light">
                저장 중…
              </Badge>
            )}
            <Button
              size="xs"
              variant={drawMode ? 'filled' : 'light'}
              onClick={() => store.setDrawMode(!drawMode)}
            >
              박스 그리기 (W)
            </Button>
          </Group>
          <Group gap="xs">
            <Tooltip label="단축키: A">
              <Button color="green" size="xs" onClick={() => approve.mutate()} loading={approve.isPending}>
                승인 (A)
              </Button>
            </Tooltip>
            <Tooltip label="단축키: X">
              <Button color="red" variant="light" size="xs" onClick={() => reject.mutate()} loading={reject.isPending}>
                반려 (X)
              </Button>
            </Tooltip>
          </Group>
        </Group>
        <div ref={canvasBox} style={{ flex: 1, borderRadius: 8, overflow: 'hidden' }}>
          {item.data?.image_url && (
            <BBoxCanvas imageUrl={item.data.image_url} width={canvasSize.w} height={canvasSize.h} />
          )}
        </div>
        <Text size="xs" c="dimmed">
          <Kbd>A</Kbd> 승인 · <Kbd>X</Kbd> 반려 · <Kbd>W</Kbd> 박스 그리기 · <Kbd>D</Kbd> 삭제 ·{' '}
          <Kbd>1-9</Kbd> 클래스 변경 · <Kbd>←→</Kbd> 이동 · <Kbd>⌘Z</Kbd> 되돌리기 · 휠 줌 · 빈 곳 드래그 팬
        </Text>
      </Stack>

      {/* right: inspector */}
      <Card withBorder w={260} p="xs">
        <Stack gap="xs">
          <Text fw={600} size="sm">
            박스 목록 ({boxes.length})
          </Text>
          <ScrollArea h={boxes.length > 6 ? 220 : undefined}>
            <Stack gap={4}>
              {boxes.map((b) => {
                const cls = item.data?.classes.find((c) => c.id === b.cls)
                return (
                  <Card
                    key={b.id}
                    p={6}
                    withBorder
                    style={{
                      cursor: 'pointer',
                      borderColor: b.id === selectedId ? classColor(b.cls) : undefined,
                    }}
                    onClick={() => store.select(b.id)}
                  >
                    <Group gap={6} wrap="nowrap">
                      <div
                        style={{
                          width: 10,
                          height: 10,
                          borderRadius: 2,
                          background: classColor(b.cls),
                          flexShrink: 0,
                        }}
                      />
                      <Text size="xs" truncate style={{ flex: 1 }}>
                        {cls?.name ?? b.cls}
                      </Text>
                      {b.score != null && (
                        <Text size="xs" c="dimmed">
                          {(b.score * 100).toFixed(0)}%
                        </Text>
                      )}
                      {b.status === 'needs_review' && (
                        <Badge size="xs" color="orange" variant="light">
                          {b.reason ?? '확인'}
                        </Badge>
                      )}
                    </Group>
                  </Card>
                )
              })}
            </Stack>
          </ScrollArea>

          {selectedBox && (
            <>
              <Text fw={600} size="sm">
                선택된 박스
              </Text>
              <Select
                label="클래스"
                searchable
                data={classOptions}
                value={String(selectedBox.cls)}
                onChange={(v) => v != null && store.updateBox(selectedBox.id, { cls: Number(v) })}
              />
              {selectedBox.sources && selectedBox.sources.length > 0 && (
                <div>
                  <Text size="xs" c="dimmed" mb={4}>
                    모델별 신뢰도
                  </Text>
                  {selectedBox.sources.map((s) => (
                    <Group key={s.model} justify="space-between">
                      <Text size="xs">{s.model}</Text>
                      <Text size="xs" c="dimmed">
                        {(s.score * 100).toFixed(0)}%
                      </Text>
                    </Group>
                  ))}
                </div>
              )}
              <Button
                size="xs"
                color="red"
                variant="light"
                onClick={() => store.deleteBox(selectedBox.id)}
              >
                박스 삭제 (D)
              </Button>
            </>
          )}

          {!selectedBox && (
            <>
              <Text fw={600} size="sm">
                새 박스 클래스
              </Text>
              <Select
                searchable
                data={classOptions}
                value={String(store.activeCls)}
                onChange={(v) => v != null && store.setActiveCls(Number(v))}
              />
            </>
          )}
        </Stack>
      </Card>
    </Group>
  )
}

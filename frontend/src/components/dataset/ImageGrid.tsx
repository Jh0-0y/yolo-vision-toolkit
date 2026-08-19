import { useEffect, useRef, useState } from 'react'
import { Badge, Checkbox, SimpleGrid, Stack, Text, Tooltip } from '@mantine/core'
import type { DatasetImage, LabelBox } from '../../api/client'
import { classColor } from '../../stores/editorStore'

/** split 배지 색 — 상단 수치 타일과 같은 색을 쓴다. */
const SPLIT_COLOR: Record<string, string> = {
  train: 'blue',
  val: 'grape',
  test: 'pink',
}

interface Props {
  items: DatasetImage[]
  selected: Set<string>
  onSelectedChange: (next: Set<string>) => void
  onOpen: (item: DatasetImage) => void
}

interface DragRect {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** Nearest scrollable ancestor, or the document scroller (window) as fallback. */
function getScrollParent(el: HTMLElement): HTMLElement {
  let n: HTMLElement | null = el.parentElement
  while (n) {
    const oy = getComputedStyle(n).overflowY
    if (/(auto|scroll|overlay)/.test(oy) && n.scrollHeight > n.clientHeight) return n
    n = n.parentElement
  }
  return (document.scrollingElement as HTMLElement | null) ?? document.documentElement
}

/** Thumbnail grid with checkbox selection, shift-click range select and
 *  rubber-band drag selection (with edge auto-scroll). */
export default function ImageGrid({ items, selected, onSelectedChange, onOpen }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cardRefs = useRef(new Map<string, HTMLDivElement>())
  const lastClicked = useRef<number | null>(null)
  const didDrag = useRef(false)
  const [drag, setDrag] = useState<DragRect | null>(null)

  const toggle = (name: string, index: number, shiftKey: boolean) => {
    const next = new Set(selected)
    if (shiftKey && lastClicked.current != null) {
      const [from, to] = [Math.min(lastClicked.current, index), Math.max(lastClicked.current, index)]
      const adding = !selected.has(name)
      for (let i = from; i <= to; i++) {
        if (adding) next.add(items[i].name)
        else next.delete(items[i].name)
      }
    } else if (next.has(name)) {
      next.delete(name)
    } else {
      next.add(name)
    }
    lastClicked.current = index
    onSelectedChange(next)
  }

  // rubber-band selection can start anywhere on the page background (including
  // the margins outside the 1200px column) — not just inside the grid. Dragging
  // the pointer to the top/bottom edge auto-scrolls so the selection can extend
  // past the visible area. All geometry is in the container's content space
  // (clientXY − live container rect), which stays fixed as the page scrolls.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (e.button !== 0) return
      const t = e.target as HTMLElement
      if (!t.closest?.('main')) return // sidebar, modals, notifications
      if (t.closest('button, input, a, select, textarea, label, [role="dialog"], [data-no-drag]'))
        return
      const container = containerRef.current
      if (!container) return
      const scroller = getScrollParent(container)
      const down = container.getBoundingClientRect()
      const startX = e.clientX - down.left
      const startY = e.clientY - down.top // content-space anchor (scroll-invariant)
      didDrag.current = false
      const additive = e.ctrlKey || e.metaKey || e.shiftKey
      const base = additive ? new Set(selected) : selected
      const prevUserSelect = document.body.style.userSelect
      let last = { x: e.clientX, y: e.clientY }
      let edgeSpeed = 0 // px/frame; sign = scroll direction
      let raf = 0

      // recompute the marquee + selection from the current pointer + scroll
      const apply = () => {
        const rect = container.getBoundingClientRect()
        const curX = last.x - rect.left
        const curY = last.y - rect.top
        if (!didDrag.current && Math.abs(curX - startX) < 5 && Math.abs(curY - startY) < 5) return
        didDrag.current = true
        document.body.style.userSelect = 'none'
        const dr = { x1: startX, y1: startY, x2: curX, y2: curY }
        setDrag(dr)

        const [lx, rx] = [Math.min(dr.x1, dr.x2), Math.max(dr.x1, dr.x2)]
        const [ty, by] = [Math.min(dr.y1, dr.y2), Math.max(dr.y1, dr.y2)]
        const next = new Set(additive ? base : [])
        for (const [name, el] of cardRefs.current) {
          const r = el.getBoundingClientRect()
          const ex1 = r.left - rect.left
          const ey1 = r.top - rect.top
          if (ex1 < rx && ex1 + r.width > lx && ey1 < by && ey1 + r.height > ty) next.add(name)
        }
        onSelectedChange(next)
      }

      // viewport band that triggers auto-scroll (window vs a scrollable ancestor)
      const edges = () => {
        if (scroller === document.scrollingElement) return { top: 0, bottom: window.innerHeight }
        const r = (scroller as HTMLElement).getBoundingClientRect()
        return { top: r.top, bottom: r.bottom }
      }

      const loop = () => {
        if (edgeSpeed !== 0) {
          scroller.scrollTop += edgeSpeed // browser clamps at the ends
          apply()
        }
        raf = requestAnimationFrame(loop)
      }
      raf = requestAnimationFrame(loop)

      const onMove = (me: MouseEvent) => {
        last = { x: me.clientX, y: me.clientY }
        apply()
        const EDGE = 64 // px band near each edge
        const MAX = 22 // px/frame at the very edge
        const { top, bottom } = edges()
        if (me.clientY < top + EDGE) {
          edgeSpeed = -Math.ceil(((top + EDGE - me.clientY) / EDGE) * MAX)
        } else if (me.clientY > bottom - EDGE) {
          edgeSpeed = Math.ceil(((me.clientY - (bottom - EDGE)) / EDGE) * MAX)
        } else {
          edgeSpeed = 0
        }
      }
      const onUp = () => {
        cancelAnimationFrame(raf)
        window.removeEventListener('mousemove', onMove)
        window.removeEventListener('mouseup', onUp)
        document.body.style.userSelect = prevUserSelect
        setDrag(null)
        // let the click event (fired right after mouseup) know a drag happened
        setTimeout(() => {
          didDrag.current = false
        }, 0)
      }
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseup', onUp)
    }
    window.addEventListener('mousedown', onDown)
    return () => window.removeEventListener('mousedown', onDown)
  }, [selected, onSelectedChange])

  return (
    <div ref={containerRef} style={{ position: 'relative', userSelect: 'none' }}>
      <SimpleGrid cols={{ base: 2, sm: 4, lg: 6 }} spacing="xs">
        {items.map((img, index) => (
          <Stack
            key={img.name}
            gap={2}
            ref={(el) => {
              if (el) cardRefs.current.set(img.name, el)
              else cardRefs.current.delete(img.name)
            }}
          >
            <div style={{ position: 'relative' }}>
              <Thumb
                img={img}
                onClick={(e) => {
                  if (didDrag.current) return
                  if (e.shiftKey || e.ctrlKey || e.metaKey) {
                    toggle(img.name, index, e.shiftKey)
                  } else {
                    onOpen(img)
                  }
                }}
                selected={selected.has(img.name)}
              />
              <Checkbox
                checked={selected.has(img.name)}
                onChange={() => {}}
                onClick={(e) => {
                  e.stopPropagation()
                  toggle(img.name, index, (e as React.MouseEvent).shiftKey)
                }}
                size="sm"
                style={{ position: 'absolute', top: 6, left: 6 }}
              />
              <div style={{ position: 'absolute', top: 6, right: 6, display: 'flex', gap: 4 }}>
                {img.labeled && img.boxes.length > 0 && (
                  <Badge size="xs" variant="filled" color="blue">
                    L
                  </Badge>
                )}
                {img.labeled && img.boxes.length === 0 && (
                  <Tooltip label="No class — label file has no boxes (negative sample)">
                    <Badge size="xs" variant="filled" color="gray">
                      N
                    </Badge>
                  </Tooltip>
                )}
                {/* 검수 여부 배지는 없다 — 탭이 이미 그것으로 갈라져 있어
                    모든 카드에 같은 표시가 붙을 뿐이다. 대신 **어느 split 인지**를
                    보인다. 색은 상단 수치 타일과 같다. */}
                {img.split && (
                  <Badge size="xs" variant="filled" color={SPLIT_COLOR[img.split]}>
                    {img.split}
                  </Badge>
                )}
              </div>
            </div>
            <Text size="xs" c="dimmed" truncate>
              {img.name}
            </Text>
          </Stack>
        ))}
      </SimpleGrid>

      {drag && (
        <div
          style={{
            position: 'absolute',
            left: Math.min(drag.x1, drag.x2),
            top: Math.min(drag.y1, drag.y2),
            width: Math.abs(drag.x2 - drag.x1),
            height: Math.abs(drag.y2 - drag.y1),
            border: '1px solid var(--mantine-color-blue-5)',
            background: 'rgba(51, 154, 240, 0.12)',
            pointerEvents: 'none',
            zIndex: 5,
          }}
        />
      )}
    </div>
  )
}

/** Square thumbnail with normalized label boxes overlaid, exactly aligned to the
 *  contained image regardless of aspect ratio. */
function Thumb({
  img,
  onClick,
  selected,
}: {
  img: { thumb: string; boxes: LabelBox[] }
  onClick: (e: React.MouseEvent) => void
  selected: boolean
}) {
  const [aspect, setAspect] = useState<number | null>(null)
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
        outline: selected ? '2px solid var(--mantine-color-blue-5)' : undefined,
        outlineOffset: -2,
      }}
    >
      <img
        src={img.thumb}
        loading="lazy"
        draggable={false}
        onLoad={(e) => setAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
        style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
      />
      {rect && img.boxes.length > 0 && (
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
          {img.boxes.map((b, i) => (
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

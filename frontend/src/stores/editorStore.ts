import { create } from 'zustand'

export interface EditorBox {
  id: string
  cls: number
  xyxy_n: [number, number, number, number]
  score?: number | null
  status?: string | null
  reason?: string | null
  sources?: { model: string; score: number }[] | null
}

export type EditorTool = 'select' | 'draw'

interface EditorState {
  boxes: EditorBox[]
  selectedId: string | null
  tool: EditorTool
  activeCls: number
  dirty: boolean
  past: EditorBox[][]
  future: EditorBox[][]
  clipboard: EditorBox[]

  load: (boxes: EditorBox[]) => void
  select: (id: string | null) => void
  setTool: (tool: EditorTool) => void
  setActiveCls: (cls: number) => void
  updateBox: (id: string, patch: Partial<EditorBox>) => void
  addBox: (box: EditorBox) => void
  deleteBox: (id: string) => void
  copy: () => void
  paste: () => void
  undo: () => void
  redo: () => void
  markSaved: () => void
}

let boxSeq = 0
export const newBoxId = () => `n${Date.now().toString(36)}_${boxSeq++}`

export const useEditorStore = create<EditorState>((set, get) => {
  const push = (next: EditorBox[]) =>
    set((s) => ({ boxes: next, past: [...s.past, s.boxes].slice(-50), future: [], dirty: true }))

  return {
    boxes: [],
    selectedId: null,
    tool: 'select',
    activeCls: 0,
    dirty: false,
    past: [],
    future: [],
    clipboard: [],

    load: (boxes) =>
      set({ boxes, selectedId: null, dirty: false, past: [], future: [] }),
    select: (id) => set({ selectedId: id }),
    setTool: (tool) => set({ tool, selectedId: tool === 'draw' ? null : get().selectedId }),
    setActiveCls: (cls) => set({ activeCls: cls }),
    updateBox: (id, patch) =>
      push(get().boxes.map((b) => (b.id === id ? { ...b, ...patch, status: 'edited' } : b))),
    addBox: (box) => {
      push([...get().boxes, box])
      set({ selectedId: box.id })
    },
    deleteBox: (id) => {
      push(get().boxes.filter((b) => b.id !== id))
      if (get().selectedId === id) set({ selectedId: null })
    },
    copy: () => {
      const { boxes, selectedId } = get()
      const sel = boxes.find((b) => b.id === selectedId)
      if (sel) set({ clipboard: [sel] })
    },
    paste: () => {
      const { clipboard, boxes } = get()
      if (!clipboard.length) return
      const OFFSET = 0.02
      const shifted = clipboard.map((b): EditorBox => {
        const [x1, y1, x2, y2] = b.xyxy_n
        const dx = Math.max(0, Math.min(OFFSET, 1 - Math.max(x1, x2)))
        const dy = Math.max(0, Math.min(OFFSET, 1 - Math.max(y1, y2)))
        // a pasted box is a fresh manual box: keep class/size, drop model metadata
        return {
          ...b,
          id: newBoxId(),
          xyxy_n: [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
          score: null,
          status: 'edited',
        }
      })
      push([...boxes, ...shifted])
      set({ selectedId: shifted[shifted.length - 1].id })
    },
    undo: () =>
      set((s) => {
        if (!s.past.length) return s
        const prev = s.past[s.past.length - 1]
        return {
          boxes: prev,
          past: s.past.slice(0, -1),
          future: [s.boxes, ...s.future],
          dirty: true,
          selectedId: null,
        }
      }),
    redo: () =>
      set((s) => {
        if (!s.future.length) return s
        const [next, ...rest] = s.future
        return {
          boxes: next,
          past: [...s.past, s.boxes],
          future: rest,
          dirty: true,
          selectedId: null,
        }
      }),
    markSaved: () => set({ dirty: false }),
  }
})

export const classColor = (cls: number) => `hsl(${(cls * 137.508) % 360}, 75%, 55%)`
// valid translucent fill — never append alpha to hsl() strings (that yields an
// invalid color that Konva renders as solid black)
export const classColorAlpha = (cls: number, alpha: number) =>
  `hsla(${(cls * 137.508) % 360}, 75%, 55%, ${alpha})`

if (import.meta.env.DEV) {
  // debugging aid: window.__editor.getState() in the browser console
  ;(window as unknown as Record<string, unknown>).__editor = useEditorStore
}

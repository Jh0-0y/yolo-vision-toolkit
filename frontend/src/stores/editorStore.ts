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

interface EditorState {
  boxes: EditorBox[]
  selectedId: string | null
  drawMode: boolean
  activeCls: number
  dirty: boolean
  past: EditorBox[][]
  future: EditorBox[][]

  load: (boxes: EditorBox[]) => void
  select: (id: string | null) => void
  setDrawMode: (on: boolean) => void
  setActiveCls: (cls: number) => void
  updateBox: (id: string, patch: Partial<EditorBox>) => void
  addBox: (box: EditorBox) => void
  deleteBox: (id: string) => void
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
    drawMode: false,
    activeCls: 0,
    dirty: false,
    past: [],
    future: [],

    load: (boxes) =>
      set({ boxes, selectedId: null, drawMode: false, dirty: false, past: [], future: [] }),
    select: (id) => set({ selectedId: id }),
    setDrawMode: (on) => set({ drawMode: on, selectedId: on ? null : get().selectedId }),
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

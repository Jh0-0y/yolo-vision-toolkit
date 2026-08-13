// Crop Draw 세션을 브라우저에 남긴다 — 탭을 옮기거나 새로고침해도 돌아올 수 있게.
//
// 여기 담기는 것은 **어디를 보고 있었는지**뿐이다: 서버 캐시 id 와 그때 돌려 둔 노브.
// 검출 결과·궤적은 담지 않는다 — `detect_id` 하나면 서버에서 다시 받아올 수 있고,
// 그 캐시가 아직 살아 있는지는 서버만 안다(`getLiveStatus`).
//
// 랩은 노브를 계속 돌려 보는 화면이라 세션을 산출물로 저장하지는 않는다. 남기고
// 싶은 결과는 crop.json 을 받거나 Crop 탭에서 크롭 런으로 돌린다.
import type { TrackcropOverrides } from '../../api/client'
import { newEntry, type DetectorEntry } from './DetectionSettings'

export interface DrawToggles {
  objInference: boolean
  showTrails: boolean
  drawCropBox: boolean
  showDeadZone: boolean
  showCenterLine: boolean
  showHighlight: boolean
}

export interface DrawSession {
  projectId: string
  detectId: string | null
  fileName: string | null
  /** 이 검출을 돌릴 때의 detection 설정 지문 — 노브가 바뀌었는지 판단한다 */
  analyzedKey: string | null
  entries: DetectorEntry[]
  sampling: number | ''
  overrides: TrackcropOverrides
  toggles: DrawToggles
}

const LS_KEY = 'yvt.crop.draw'

export const DEFAULT_TOGGLES: DrawToggles = {
  objInference: true,
  showTrails: true,
  drawCropBox: true,
  showDeadZone: true,
  showCenterLine: true,
  showHighlight: true,
}

export function emptySession(projectId: string): DrawSession {
  return {
    projectId,
    detectId: null,
    fileName: null,
    analyzedKey: null,
    entries: [newEntry('full')],
    sampling: '',
    overrides: {},
    toggles: { ...DEFAULT_TOGGLES },
  }
}

/** 이 프로젝트의 마지막 세션. 없거나 다른 프로젝트의 것이면 빈 세션을 준다. */
export function loadSession(projectId: string): DrawSession {
  try {
    const raw = localStorage.getItem(LS_KEY)
    if (!raw) return emptySession(projectId)
    const saved = JSON.parse(raw) as Partial<DrawSession>
    if (saved.projectId !== projectId) return emptySession(projectId)
    const base = emptySession(projectId)
    return {
      ...base,
      ...saved,
      projectId,
      // 저장된 뒤에 노브가 늘어났을 수 있으므로 기본값 위에 얹는다
      toggles: { ...base.toggles, ...(saved.toggles ?? {}) },
      entries: saved.entries?.length ? saved.entries : base.entries,
    }
  } catch {
    return emptySession(projectId) // quota / 손상된 값 — 세션 없는 셈 친다
  }
}

export function saveSession(session: DrawSession): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(session))
  } catch {
    // best-effort — quota / disabled storage
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(LS_KEY)
  } catch {
    // best-effort
  }
}

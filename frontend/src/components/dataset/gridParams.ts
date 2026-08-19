// 그리드와 라벨 에디터가 **같은 목록을 보게 하는** URL 파라미터 계약.
//
// 에디터의 `3 / 128` 과 이전·다음은 그리드에서 좁혀 놓은 목록을 따라야 한다. 그러려면
// 필터가 컴포넌트 안이 아니라 **URL 에** 있어야 한다 — 그래야 링크를 타고 넘어가고,
// 새로고침해도 남고, 뒤로 가기로 돌아온다.
//
// 키 이름을 여기 한 번만 적는다. 양쪽이 따로 적으면 한쪽만 고쳐져 조용히 어긋난다.
import type { DatasetImageQuery } from '../../api/client'
import { DEFAULT_FILTERS, type GridFilterState } from './GridFilters'

export const SPLIT_ALL = 'all'

export function readFilters(sp: URLSearchParams): GridFilterState {
  const cls = sp.get('cls')
  const sort = sp.get('sort')
  const order = sp.get('order')
  return {
    q: sp.get('q') ?? DEFAULT_FILTERS.q,
    cls: cls === null || cls === '' ? null : Number(cls),
    sort: sort === 'name' ? 'name' : 'created',
    order: order === 'asc' ? 'asc' : 'desc',
  }
}

export function readSplit(sp: URLSearchParams): string {
  return sp.get('split') ?? SPLIT_ALL
}

export function readPage(sp: URLSearchParams): number {
  const n = Number(sp.get('page'))
  return Number.isFinite(n) && n >= 1 ? n : 1
}

/** 기본값인 항목은 URL 에 **쓰지 않는다** — 주소가 짧아야 읽을 만하다. */
function put(sp: URLSearchParams, key: string, value: string, fallback: string) {
  if (value === fallback) sp.delete(key)
  else sp.set(key, value)
}

/** `tab` 처럼 우리가 모르는 파라미터는 건드리지 않고 남긴다. */
export function writeGridParams(
  sp: URLSearchParams,
  next: { filters?: GridFilterState; split?: string; page?: number },
): URLSearchParams {
  const out = new URLSearchParams(sp)
  if (next.filters) {
    const f = next.filters
    put(out, 'q', f.q, '')
    put(out, 'cls', f.cls === null ? '' : String(f.cls), '')
    put(out, 'sort', f.sort, DEFAULT_FILTERS.sort)
    put(out, 'order', f.order, DEFAULT_FILTERS.order)
  }
  if (next.split !== undefined) put(out, 'split', next.split, SPLIT_ALL)
  if (next.page !== undefined) put(out, 'page', String(next.page), '1')
  return out
}

/** 에디터로 넘길 검색 문자열. `reviewed` 는 에디터가 3값(`yes`/`no`/`all`)으로 읽는다. */
export function editorSearch(
  filters: GridFilterState,
  reviewed: boolean,
  split: string,
): string {
  const sp = new URLSearchParams()
  sp.set('reviewed', reviewed ? 'yes' : 'no')
  if (filters.q) sp.set('q', filters.q)
  if (filters.cls !== null) sp.set('cls', String(filters.cls))
  if (filters.sort !== DEFAULT_FILTERS.sort) sp.set('sort', filters.sort)
  if (filters.order !== DEFAULT_FILTERS.order) sp.set('order', filters.order)
  if (reviewed && split !== SPLIT_ALL) sp.set('split', split)
  return sp.toString()
}

/** 목록 요청 하나로 모으기 — 그리드와 에디터가 **같은 조건**을 보내야 순서가 같다. */
export function toImageQuery(
  filters: GridFilterState,
  reviewed: boolean,
  split: string,
): DatasetImageQuery {
  return {
    reviewed,
    sort: filters.sort,
    order: filters.order,
    ...(filters.q ? { q: filters.q } : {}),
    ...(filters.cls !== null ? { cls: filters.cls } : {}),
    ...(reviewed && split !== SPLIT_ALL
      ? { split: split as DatasetImageQuery['split'] }
      : {}),
  }
}

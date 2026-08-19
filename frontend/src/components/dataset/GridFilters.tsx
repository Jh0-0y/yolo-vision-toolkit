// 그리드 위의 필터 줄 — 이름 검색 · 클래스 · 정렬. 두 탭이 같은 것을 쓴다.
//
// 클래스 필터에는 **`No class`** 가 있다. 라벨 파일은 있는데 박스가 없는 이미지로,
// "아직 안 그렸다"가 아니라 **"여기엔 없다"는 학습 신호**다. 서버는 이것을 `cls=-1`
// 로 받는다.
import { Group, Select, TextInput } from '@mantine/core'
import { IconSearch } from '@tabler/icons-react'

/** 클래스 필터의 특수값 — 서버의 `cls=-1` 과 짝이다. */
export const NO_CLASS = -1

export interface GridFilterState {
  q: string
  cls: number | null
  sort: 'created' | 'name'
  order: 'asc' | 'desc'
}

export const DEFAULT_FILTERS: GridFilterState = {
  q: '',
  cls: null,
  sort: 'created',
  order: 'desc',
}

interface Props {
  value: GridFilterState
  onChange: (next: GridFilterState) => void
  classes: { id: number; name: string }[]
}

const SORTS = [
  { value: 'created:desc', label: 'Newest first' },
  { value: 'created:asc', label: 'Oldest first' },
  { value: 'name:asc', label: 'Name A→Z' },
  { value: 'name:desc', label: 'Name Z→A' },
]

export default function GridFilters({ value, onChange, classes }: Props) {
  const classOptions = [
    { value: 'all', label: 'All classes' },
    { value: String(NO_CLASS), label: 'No class (negative)' },
    ...classes.map((c) => ({ value: String(c.id), label: c.name })),
  ]

  return (
    <Group gap="xs">
      <TextInput
        placeholder="Search by name"
        leftSection={<IconSearch size={14} />}
        value={value.q}
        onChange={(e) => onChange({ ...value, q: e.currentTarget.value })}
        w={220}
      />
      <Select
        w={190}
        data={classOptions}
        value={value.cls === null ? 'all' : String(value.cls)}
        onChange={(v) => onChange({ ...value, cls: v === 'all' || v === null ? null : Number(v) })}
        allowDeselect={false}
        comboboxProps={{ withinPortal: true }}
      />
      <Select
        w={150}
        data={SORTS}
        value={`${value.sort}:${value.order}`}
        onChange={(v) => {
          const [sort, order] = (v ?? 'created:desc').split(':')
          onChange({
            ...value,
            sort: sort as GridFilterState['sort'],
            order: order as GridFilterState['order'],
          })
        }}
        allowDeselect={false}
        comboboxProps={{ withinPortal: true }}
      />
    </Group>
  )
}

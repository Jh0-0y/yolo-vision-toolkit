// 그리드 위의 필터 줄 — 이름 검색 · 클래스 · 정렬. 두 탭이 같은 것을 쓴다.
//
// 클래스 필터에는 **`No class`** 가 있다. 라벨 파일은 있는데 박스가 없는 이미지로,
// "아직 안 그렸다"가 아니라 **"여기엔 없다"는 학습 신호**다. 서버는 이것을 `cls=-1`
// 로 받는다.
//
// 클래스 점 색은 **라벨 박스와 같은 색**이다(`classColor`) — 목록에서 고른 색이
// 썸네일 위 박스 색과 같아야 눈이 헤매지 않는다.
import { ActionIcon, Group, Select, Text, TextInput } from '@mantine/core'
import {
  IconArrowsSort,
  IconSearch,
  IconSortAscendingLetters,
  IconSortDescendingLetters,
  IconTag,
  IconX,
} from '@tabler/icons-react'
import { classColor } from '../../stores/editorStore'

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
  { value: 'name:asc', label: 'Name A–Z' },
  { value: 'name:desc', label: 'Name Z–A' },
]

const SORT_ICON: Record<string, React.ReactNode> = {
  'created:desc': <IconArrowsSort size={16} />,
  'created:asc': <IconArrowsSort size={16} />,
  'name:asc': <IconSortAscendingLetters size={16} />,
  'name:desc': <IconSortDescendingLetters size={16} />,
}

function Dot({ color }: { color: string }) {
  return (
    <span
      style={{
        width: 10,
        height: 10,
        borderRadius: '50%',
        background: color,
        display: 'inline-block',
        flexShrink: 0,
      }}
    />
  )
}

export default function GridFilters({ value, onChange, classes }: Props) {
  const classOptions = [
    { value: 'all', label: 'All classes' },
    { value: String(NO_CLASS), label: 'No class' },
    ...classes.map((c) => ({ value: String(c.id), label: c.name })),
  ]
  const sortValue = `${value.sort}:${value.order}`
  const dirty = value.q !== '' || value.cls !== null || sortValue !== 'created:desc'

  // 고른 클래스가 있으면 아이콘 자리에 **그 색 점**을 둔다 — 무엇으로 좁혀 놓았는지
  // 열어 보지 않아도 보인다.
  const classLeft =
    value.cls === null ? (
      <IconTag size={16} />
    ) : value.cls === NO_CLASS ? (
      <Dot color="var(--mantine-color-gray-5)" />
    ) : (
      <Dot color={classColor(value.cls)} />
    )

  return (
    <Group gap="xs" wrap="wrap">
      <TextInput
        placeholder="Search by name"
        leftSection={<IconSearch size={16} />}
        rightSection={
          value.q ? (
            <ActionIcon
              variant="subtle"
              color="gray"
              size="sm"
              aria-label="Clear search"
              onClick={() => onChange({ ...value, q: '' })}
            >
              <IconX size={14} />
            </ActionIcon>
          ) : null
        }
        value={value.q}
        onChange={(e) => onChange({ ...value, q: e.currentTarget.value })}
        w={230}
      />

      <Select
        w={200}
        aria-label="Filter by class"
        leftSection={classLeft}
        data={classOptions}
        value={value.cls === null ? 'all' : String(value.cls)}
        onChange={(v) => onChange({ ...value, cls: v === 'all' || v === null ? null : Number(v) })}
        allowDeselect={false}
        comboboxProps={{ withinPortal: true }}
        renderOption={({ option }) => (
          <Group gap="xs" wrap="nowrap">
            {option.value === 'all' ? (
              <IconTag size={14} />
            ) : option.value === String(NO_CLASS) ? (
              <Dot color="var(--mantine-color-gray-5)" />
            ) : (
              <Dot color={classColor(Number(option.value))} />
            )}
            <div style={{ minWidth: 0 }}>
              <Text size="sm" truncate>
                {option.label}
              </Text>
              {option.value === String(NO_CLASS) && (
                <Text size="xs" c="dimmed">
                  Empty label — a negative sample
                </Text>
              )}
            </div>
          </Group>
        )}
      />

      <Select
        w={170}
        aria-label="Sort"
        leftSection={SORT_ICON[sortValue]}
        data={SORTS}
        value={sortValue}
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
        renderOption={({ option }) => (
          <Group gap="xs" wrap="nowrap">
            {SORT_ICON[option.value]}
            <Text size="sm">{option.label}</Text>
          </Group>
        )}
      />

      {/* 좁혀 놓은 것이 있을 때만 보인다 — 평소에 자리를 차지하지 않는다 */}
      {dirty && (
        <ActionIcon
          variant="subtle"
          color="gray"
          size="lg"
          aria-label="Reset filters"
          title="Reset filters"
          onClick={() => onChange(DEFAULT_FILTERS)}
        >
          <IconX size={16} />
        </ActionIcon>
      )}
    </Group>
  )
}

// 그리드 아래에 붙는 선택·일괄작업 바.
//
// **`Select all` 은 보이는 페이지가 아니라 필터에 걸린 전부를 고른다.** 페이지가
// 60장 단위라, 이게 없으면 수백 장을 넘기는 일이 페이지를 돌며 손으로 고르는 일이 된다.
//
// 내보내기 버튼은 없다 — 내보내기는 선택이 아니라 **데이터셋 단위**의 일이라
// `ExportCard` 가 맡는다.
import { Button, Card, Group, Text } from '@mantine/core'
import {
  IconArrowBackUp,
  IconChecks,
  IconSquareOff,
  IconTrash,
  IconWand,
} from '@tabler/icons-react'

interface Props {
  /** 이 패널이 보여주는 쪽 — 검수 버튼의 방향이 갈린다 */
  reviewed: boolean
  selectedCount: number
  total: number
  selectingAll?: boolean
  reviewPending?: boolean
  deletePending?: boolean
  onSelectAll: () => void
  onClear: () => void
  onAutoLabel: () => void
  onReview: () => void
  onDelete: () => void
}

export default function SelectionBar({
  reviewed,
  selectedCount,
  total,
  selectingAll,
  reviewPending,
  deletePending,
  onSelectAll,
  onClear,
  onAutoLabel,
  onReview,
  onDelete,
}: Props) {
  const none = selectedCount === 0
  return (
    <Card
      withBorder
      radius="md"
      padding="sm"
      data-no-drag
      style={{ position: 'sticky', bottom: 12, zIndex: 10, boxShadow: 'var(--mantine-shadow-md)' }}
    >
      <Group justify="space-between" wrap="wrap">
        <Group gap="xs">
          <Button
            variant="subtle"
            size="compact-sm"
            leftSection={<IconChecks size={14} />}
            onClick={onSelectAll}
            loading={selectingAll}
            disabled={total === 0}
          >
            Select all ({total})
          </Button>
          {selectedCount > 0 && (
            <>
              <Text size="sm" fw={600}>
                {selectedCount} selected
              </Text>
              <Button
                variant="subtle"
                size="compact-sm"
                color="gray"
                leftSection={<IconSquareOff size={14} />}
                onClick={onClear}
              >
                Clear
              </Button>
            </>
          )}
        </Group>

        <Group gap="xs">
          {/* 오토라벨링은 미검수 탭의 일이다. 선택이 없으면 **전체**가 대상이라
              비활성화하지 않는다 — 문구로 대상을 밝힌다. */}
          {!reviewed && (
            <Button variant="light" leftSection={<IconWand size={16} />} onClick={onAutoLabel}>
              {none ? 'Auto-label all' : `Auto-label (${selectedCount})`}
            </Button>
          )}
          <Button
            leftSection={reviewed ? <IconArrowBackUp size={16} /> : <IconChecks size={16} />}
            onClick={onReview}
            loading={reviewPending}
            disabled={none}
          >
            {reviewed ? 'Send back' : 'Mark reviewed'}
          </Button>
          <Button
            variant="light"
            color="red"
            leftSection={<IconTrash size={16} />}
            onClick={onDelete}
            loading={deletePending}
            disabled={none}
          >
            Delete
          </Button>
        </Group>
      </Group>
    </Card>
  )
}

import { Button, Card, Group, Text } from '@mantine/core'
import { IconChecks, IconDownload, IconSquareOff, IconTrash, IconWand } from '@tabler/icons-react'

interface Props {
  selectedCount: number
  total: number
  onSelectAll: () => void
  onClear: () => void
  onAutoLabel: () => void
  onExport: () => void
  onDelete: () => void
}

/** Sticky control bar under the grid: selection controls + bulk actions.
 *  Actions require a selection — use Select all to target everything. */
export default function SelectionBar({
  selectedCount,
  total,
  onSelectAll,
  onClear,
  onAutoLabel,
  onExport,
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
            disabled={total === 0}
          >
            Select all
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
          <Button leftSection={<IconWand size={16} />} onClick={onAutoLabel} disabled={none}>
            Auto Labeling
          </Button>
          <Button
            variant="light"
            leftSection={<IconDownload size={16} />}
            onClick={onExport}
            disabled={none}
          >
            Export
          </Button>
          <Button
            variant="light"
            color="red"
            leftSection={<IconTrash size={16} />}
            onClick={onDelete}
            disabled={none}
          >
            Delete
          </Button>
        </Group>
      </Group>
    </Card>
  )
}

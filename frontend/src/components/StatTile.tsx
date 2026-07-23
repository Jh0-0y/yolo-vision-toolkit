import { Group, Paper, Text } from '@mantine/core'
import type { ReactNode } from 'react'

interface Props {
  label: string
  value: ReactNode
  color?: string
  icon?: ReactNode
  title?: string
  /** small secondary line under the value (e.g. a comparison value) */
  sub?: ReactNode
  style?: React.CSSProperties
}

/** Prominent count tile: a large colored number over an uppercase label. */
export default function StatTile({ label, value, color = 'gray', icon, title, sub, style }: Props) {
  return (
    <Paper
      withBorder
      radius="md"
      px="sm"
      py={6}
      title={title}
      style={{ minWidth: 84, flex: '1 1 0', ...style }}
    >
      <Group gap={5} wrap="nowrap" mb={2} c="dimmed">
        {icon}
        <Text size="xs" fw={600} tt="uppercase" style={{ letterSpacing: 0.4 }}>
          {label}
        </Text>
      </Group>
      <Text size="1.6rem" fw={800} lh={1.1} c={color}>
        {value}
      </Text>
      {sub != null && (
        <Text size="xs" c="dimmed" mt={2}>
          {sub}
        </Text>
      )}
    </Paper>
  )
}

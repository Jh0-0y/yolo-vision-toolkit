import {
  useState,
} from 'react'
import {
  Card,
  Group,
  Progress,
  Table,
  Text,
} from '@mantine/core'
import {
  type PerClassRow,
} from '../../api/client'
import {
  classColor,
} from '../../stores/editorStore'


type PcKey = 'name' | 'instances' | 'precision' | 'recall' | 'mAP50' | 'mAP50-95'

function mapColor(v: number): string {
  if (v < 0.3) return 'red'
  if (v < 0.5) return 'orange'
  if (v < 0.7) return 'yellow'
  return 'teal'
}

export default function PerClassTable({ rows }: { rows: PerClassRow[] }) {
  const [sortKey, setSortKey] = useState<PcKey>('mAP50-95')
  const [asc, setAsc] = useState(true) // weak classes first by default
  const hasInstances = rows.some((r) => r.instances != null)

  const sorted = [...rows].sort((a, b) => {
    const av = a[sortKey]
    const bv = b[sortKey]
    if (typeof av === 'string' || typeof bv === 'string') {
      return (asc ? 1 : -1) * String(av).localeCompare(String(bv))
    }
    return (asc ? 1 : -1) * ((av ?? -Infinity) - (bv ?? -Infinity))
  })

  const toggle = (k: PcKey) => {
    if (k === sortKey) setAsc((v) => !v)
    else {
      setSortKey(k)
      setAsc(true)
    }
  }

  const Th = ({ k, children, ta }: { k: PcKey; children: React.ReactNode; ta?: 'right' }) => (
    <Table.Th
      style={{ cursor: 'pointer', textAlign: ta }}
      onClick={() => toggle(k)}
    >
      {children}
      {sortKey === k ? (asc ? ' ▲' : ' ▼') : ''}
    </Table.Th>
  )

  return (
    <Card withBorder radius="md" padding="md">
      <Text size="sm" fw={600} mb="xs">
        Per-class metrics
      </Text>
      <Table.ScrollContainer minWidth={480}>
        <Table highlightOnHover verticalSpacing={6} horizontalSpacing="md">
          <Table.Thead>
            <Table.Tr>
              <Th k="name">Class</Th>
              {hasInstances && <Th k="instances" ta="right">Instances</Th>}
              <Th k="precision" ta="right">P</Th>
              <Th k="recall" ta="right">R</Th>
              <Th k="mAP50" ta="right">mAP50</Th>
              <Th k="mAP50-95" ta="right">mAP50-95</Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.map((r) => (
              <Table.Tr key={r.cls}>
                <Table.Td>
                  <Group gap={6} wrap="nowrap">
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 3,
                        background: classColor(r.cls),
                        flexShrink: 0,
                      }}
                    />
                    <Text size="sm" fw={500} truncate>
                      {r.name}
                    </Text>
                  </Group>
                </Table.Td>
                {hasInstances && (
                  <Table.Td ta="right">
                    <Text size="sm" c="dimmed">
                      {r.instances ?? '–'}
                    </Text>
                  </Table.Td>
                )}
                <Table.Td ta="right">{r.precision.toFixed(3)}</Table.Td>
                <Table.Td ta="right">{r.recall.toFixed(3)}</Table.Td>
                <Table.Td ta="right">{r.mAP50.toFixed(3)}</Table.Td>
                <Table.Td>
                  <Group gap="xs" wrap="nowrap" justify="flex-end">
                    <Progress
                      value={r['mAP50-95'] * 100}
                      color={mapColor(r['mAP50-95'])}
                      w={64}
                      size="sm"
                    />
                    <Text size="sm" fw={600} w={44} ta="right">
                      {r['mAP50-95'].toFixed(3)}
                    </Text>
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Card>
  )
}

// primary params stay visible; everything else hides behind the toggle

import {
  useState,
} from 'react'
import {
  Button,
  Card,
  Collapse,
  Table,
  Text,
} from '@mantine/core'
import {
  IconChevronDown,
} from '@tabler/icons-react'
import {
  type TrainRunOut,
} from '../../api/client'
import {
  formatDuration,
} from './metrics'


const PRIMARY_PARAMS = ['epochs', 'imgsz', 'batch']

export default function DetailsCard({ run, durationSec }: { run: TrainRunOut; durationSec?: number }) {
  const [showParams, setShowParams] = useState(false)
  const metaRows: [string, React.ReactNode][] = [
    ['Base model', run.base_model_name ?? run.base_model_id],
    ['Status', run.status],
    ['Started', new Date(run.created_at).toLocaleString()],
    ['Finished', run.finished_at ? new Date(run.finished_at).toLocaleString() : '–'],
    ['Duration', formatDuration(durationSec)],
    ...PRIMARY_PARAMS.filter((k) => k in run.params).map(
      (k) => [k, String(run.params[k])] as [string, React.ReactNode],
    ),
  ]
  const paramRows = Object.entries(run.params)
    .filter(([k]) => !PRIMARY_PARAMS.includes(k))
    .map(([k, v]) => [k, String(v)] as [string, React.ReactNode])
  const renderRows = (rows: [string, React.ReactNode][]) =>
    rows.map(([k, v]) => (
      <Table.Tr key={k}>
        <Table.Th w={160} style={{ color: 'var(--mantine-color-dimmed)', fontWeight: 500 }}>
          {k}
        </Table.Th>
        <Table.Td>
          <Text size="sm">{v}</Text>
        </Table.Td>
      </Table.Tr>
    ))
  return (
    <Card withBorder radius="md" padding="md">
      <Text size="sm" fw={600} mb="xs">
        Run details
      </Text>
      <Table variant="vertical" withRowBorders={false} verticalSpacing={4}>
        <Table.Tbody>{renderRows(metaRows)}</Table.Tbody>
      </Table>
      {paramRows.length > 0 && (
        <>
          <Button
            variant="subtle"
            size="compact-sm"
            mt={4}
            rightSection={<IconChevronDown size={14} />}
            onClick={() => setShowParams((v) => !v)}
          >
            {showParams ? 'Hide' : 'Show'} parameters ({paramRows.length})
          </Button>
          <Collapse expanded={showParams}>
            <Table variant="vertical" withRowBorders={false} verticalSpacing={4}>
              <Table.Tbody>{renderRows(paramRows)}</Table.Tbody>
            </Table>
          </Collapse>
        </>
      )}
    </Card>
  )
}

import { Badge, Card, Group, Stack, Table, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { api, type TrainRunOut } from '../api/client'

const RUN_STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  stopped: 'yellow',
  queued: 'gray',
}

export default function TrainingHistoryPage() {
  const navigate = useNavigate()
  const runs = useQuery({
    queryKey: ['train-runs'],
    queryFn: () => api.get<TrainRunOut[]>('/train/runs'),
    refetchInterval: 15_000,
  })

  return (
    <Stack gap="lg">
      <div>
        <Title order={3}>Training History</Title>
        <Text c="dimmed" size="sm">
          Click a run to see live progress and results.
        </Text>
      </div>

      <Card withBorder radius="md" padding="sm">
        <Table highlightOnHover verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Name</Table.Th>
              <Table.Th>Base model</Table.Th>
              <Table.Th>Config</Table.Th>
              <Table.Th>mAP50</Table.Th>
              <Table.Th>Started</Table.Th>
              <Table.Th ta="right">Status</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {runs.data?.map((r) => (
              <Table.Tr key={r.id} style={{ cursor: 'pointer' }} onClick={() => navigate(r.id)}>
                <Table.Td>
                  <Text size="sm" fw={600}>
                    {r.name}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{r.base_model_name ?? '-'}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed">
                    ep {r.params.epochs} · {r.params.imgsz}px · batch {r.params.batch}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">
                    {r.metrics?.['metrics/mAP50(B)'] != null
                      ? r.metrics['metrics/mAP50(B)'].toFixed(3)
                      : '–'}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed">
                    {new Date(r.created_at).toLocaleString()}
                  </Text>
                </Table.Td>
                <Table.Td ta="right">
                  <Group justify="flex-end">
                    <Badge color={RUN_STATUS_COLOR[r.status] ?? 'gray'} variant="light">
                      {r.status}
                    </Badge>
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
        {runs.data?.length === 0 && (
          <Text size="sm" c="dimmed" p="md">
            No training runs yet. Start one from Train.
          </Text>
        )}
      </Card>
    </Stack>
  )
}

import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconPlus, IconTrash } from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { api, type TrainRunOut } from '../api/client'

const RUN_STATUS_COLOR: Record<string, string> = {
  running: 'blue',
  done: 'green',
  error: 'red',
  stopped: 'yellow',
  queued: 'gray',
}

const TERMINAL_STATUS = new Set(['done', 'error', 'stopped'])

export default function TrainingHistoryPage() {
  const { projectId = '' } = useParams()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const runs = useQuery({
    queryKey: ['train-runs', projectId],
    queryFn: () => api.get<TrainRunOut[]>(`/training/runs?project_id=${projectId}`),
    // poll fast while something is active so the status/spinner updates promptly
    refetchInterval: (query) => {
      const data = query.state.data as TrainRunOut[] | undefined
      return data?.some((r) => !TERMINAL_STATUS.has(r.status)) ? 3_000 : 15_000
    },
  })

  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/training/runs/${id}`),
    onSuccess: () => {
      notifications.show({ message: 'Training run deleted', color: 'green' })
      qc.invalidateQueries({ queryKey: ['train-runs', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  return (
    <Stack gap="lg">
      {/* 이력이 먼저다 — 학습은 여기서 시작한다 */}
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={3}>Training History</Title>
          <Text c="dimmed" size="sm">
            Click a run to see live progress and results.
          </Text>
        </div>
        <Button leftSection={<IconPlus size={16} />} onClick={() => navigate('new')}>
          새 학습 시작
        </Button>
      </Group>

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
              <Table.Th w={48} />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {runs.data?.map((r) => (
              <Table.Tr key={r.id} style={{ cursor: 'pointer' }} onClick={() => navigate(r.id)}>
                <Table.Td>
                  <Group gap="xs" wrap="nowrap">
                    {!TERMINAL_STATUS.has(r.status) && (
                      <Loader size={14} color={RUN_STATUS_COLOR[r.status] ?? 'gray'} />
                    )}
                    <Text size="sm" fw={600}>
                      {r.name}
                    </Text>
                  </Group>
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
                <Table.Td onClick={(e) => e.stopPropagation()}>
                  <Tooltip
                    label={
                      TERMINAL_STATUS.has(r.status)
                        ? 'Delete run'
                        : 'Stop the run before deleting'
                    }
                    withArrow
                  >
                    <ActionIcon
                      variant="subtle"
                      color="red"
                      aria-label="Delete run"
                      disabled={!TERMINAL_STATUS.has(r.status)}
                      loading={remove.isPending && remove.variables === r.id}
                      onClick={() => {
                        if (confirm(`Delete training run "${r.name}"? This removes its logs and weights.`))
                          remove.mutate(r.id)
                      }}
                    >
                      <IconTrash size={16} />
                    </ActionIcon>
                  </Tooltip>
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

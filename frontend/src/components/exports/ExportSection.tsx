import { useState } from 'react'
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  Modal,
  NumberInput,
  Stack,
  Table,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type ExportOut } from '../../api/client'

const fmtSize = (bytes: number) =>
  bytes > 1 << 20 ? `${(bytes / (1 << 20)).toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`

export default function ExportSection({
  projectId,
  confirmedCount,
}: {
  projectId: string
  confirmedCount: number
}) {
  const queryClient = useQueryClient()
  const [opened, setOpened] = useState(false)
  const [valSplit, setValSplit] = useState<number | string>(0.2)
  const [seed, setSeed] = useState<number | string>(42)

  const exports = useQuery({
    queryKey: ['exports', projectId],
    queryFn: () => api.get<ExportOut[]>(`/projects/${projectId}/exports`),
  })

  const create = useMutation({
    mutationFn: () =>
      api.post<ExportOut>(`/projects/${projectId}/exports`, {
        val_split: Number(valSplit),
        seed: Number(seed),
      }),
    onSuccess: (e) => {
      notifications.show({
        message: `내보내기 완료 — train ${e.train} / val ${e.val}`,
        color: 'green',
      })
      queryClient.invalidateQueries({ queryKey: ['exports', projectId] })
      setOpened(false)
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/projects/${projectId}/exports/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['exports', projectId] }),
  })

  return (
    <Card withBorder>
      <Group justify="space-between" mb="xs">
        <Text fw={600}>데이터셋 내보내기</Text>
        <Button size="xs" onClick={() => setOpened(true)} disabled={confirmedCount === 0}>
          새 내보내기
        </Button>
      </Group>

      {exports.data?.length === 0 && (
        <Text size="sm" c="dimmed">
          확정 데이터를 train/val로 분할해 YOLO 학습용 zip(data.yaml 포함)으로 내보냅니다.
        </Text>
      )}

      {!!exports.data?.length && (
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>생성일</Table.Th>
              <Table.Th>train / val</Table.Th>
              <Table.Th>클래스</Table.Th>
              <Table.Th>크기</Table.Th>
              <Table.Th />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {exports.data.map((e) => (
              <Table.Tr key={e.id}>
                <Table.Td>
                  <Text size="sm">{new Date(e.created_at).toLocaleString()}</Text>
                </Table.Td>
                <Table.Td>
                  <Badge variant="light">
                    {e.train} / {e.val}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{e.classes}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{fmtSize(e.size_bytes)}</Text>
                </Table.Td>
                <Table.Td>
                  <Group gap="xs" justify="flex-end">
                    <Button
                      component="a"
                      href={`/api/projects/${projectId}/exports/${e.id}/download`}
                      download
                      size="xs"
                      variant="light"
                      leftSection="⬇"
                    >
                      zip 다운로드
                    </Button>
                    <ActionIcon variant="subtle" color="red" onClick={() => remove.mutate(e.id)}>
                      ✕
                    </ActionIcon>
                  </Group>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      )}

      <Modal opened={opened} onClose={() => setOpened(false)} title="데이터셋 내보내기">
        <Stack>
          <NumberInput
            label="검증(val) 비율"
            description="0이면 전체를 train으로"
            value={valSplit}
            onChange={setValSplit}
            min={0}
            max={0.9}
            step={0.05}
            decimalScale={2}
          />
          <NumberInput label="셔플 시드" value={seed} onChange={setSeed} step={1} />
          <Button onClick={() => create.mutate()} loading={create.isPending}>
            내보내기
          </Button>
        </Stack>
      </Modal>
    </Card>
  )
}

// 검수완료 이미지를 train / val / test 로 나눈다.
//
// **기본은 미할당만 나눈다.** 이미 배정된 것을 다시 섞으면 어제 test 였던 이미지가
// 오늘 train 으로 가고, 이전 평가와 비교가 성립하지 않는다. 전부 다시 섞는 것은
// 따로 눌러야 하고 그때 경고한다.
import { useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Group,
  NumberInput,
  Stack,
  Text,
  ThemeIcon,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconAlertTriangle, IconArrowsShuffle, IconChartPie } from '@tabler/icons-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { splitDataset, type DatasetOut } from '../../api/client'

interface Props {
  projectId: string
  datasetId: string
  dataset: DatasetOut
}

export default function SplitCard({ projectId, datasetId, dataset }: Props) {
  const qc = useQueryClient()
  const [train, setTrain] = useState(8)
  const [val, setVal] = useState(1)
  const [test, setTest] = useState(1)

  const run = useMutation({
    mutationFn: (reassignAll: boolean) =>
      splitDataset(projectId, datasetId, { train, val, test, reassign_all: reassignAll }),
    onSuccess: (d) => {
      notifications.show({
        message: `Split: train ${d.train} · val ${d.val} · test ${d.test}`,
        color: 'green',
      })
      qc.invalidateQueries({ queryKey: ['dataset', projectId, datasetId] })
      qc.invalidateQueries({ queryKey: ['dataset-images', projectId, datasetId] })
      qc.invalidateQueries({ queryKey: ['datasets', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const total = train + val + test
  const pct = (v: number) => (total > 0 ? Math.round((v / total) * 100) : 0)

  return (
    <Card withBorder radius="md" padding="lg">
      <Stack gap="md">
        <Group gap="xs">
          <ThemeIcon variant="light" size="lg" radius="md">
            <IconChartPie size={20} />
          </ThemeIcon>
          <div>
            <Text fw={600}>Split</Text>
            <Text size="xs" c="dimmed">
              Divide reviewed images into train / val / test
            </Text>
          </div>
        </Group>

        <Group gap="xs" align="flex-end">
          <NumberInput label="Train" value={train} onChange={(v) => setTrain(Number(v) || 0)} min={0} w={90} />
          <Text c="dimmed" pb={8}>:</Text>
          <NumberInput label="Val" value={val} onChange={(v) => setVal(Number(v) || 0)} min={0} w={90} />
          <Text c="dimmed" pb={8}>:</Text>
          <NumberInput label="Test" value={test} onChange={(v) => setTest(Number(v) || 0)} min={0} w={90} />
          <Text size="xs" c="dimmed" pb={8}>
            {pct(train)}% / {pct(val)}% / {pct(test)}%
          </Text>
        </Group>

        {dataset.unassigned > 0 ? (
          <Text size="sm">
            <strong>{dataset.unassigned}</strong> reviewed images are unassigned.
          </Text>
        ) : (
          <Text size="sm" c="dimmed">
            Everything reviewed is already assigned.
          </Text>
        )}

        <Group gap="xs">
          <Button
            leftSection={<IconChartPie size={16} />}
            loading={run.isPending && run.variables === false}
            disabled={dataset.unassigned === 0}
            onClick={() => run.mutate(false)}
          >
            Split unassigned ({dataset.unassigned})
          </Button>
          <Button
            variant="light"
            color="orange"
            leftSection={<IconArrowsShuffle size={16} />}
            loading={run.isPending && run.variables === true}
            disabled={dataset.reviewed === 0}
            onClick={() => {
              if (
                confirm(
                  'Reshuffle ALL reviewed images?\n\n' +
                    'Images already in train/val/test will move. Past evaluations will no ' +
                    'longer be comparable, because test images can end up in training.',
                )
              )
                run.mutate(true)
            }}
          >
            Reshuffle all
          </Button>
        </Group>

        {dataset.test > 0 && (
          <Alert color="orange" icon={<IconAlertTriangle size={16} />} p="xs">
            <Text size="xs">
              Reshuffling moves images out of test — only do it when starting a fresh experiment.
            </Text>
          </Alert>
        )}
      </Stack>
    </Card>
  )
}

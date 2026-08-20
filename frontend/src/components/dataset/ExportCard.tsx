// 내보내기 세 가지. **이력을 남기지 않는다** — 만들어서 받으면 끝이고, 새로
// 만들면 이전 zip 은 사라진다. 목록이 필요하면 데이터셋 자체가 목록이다.
//
// 이미지는 하드링크로 펼쳐지므로 몇 번을 내보내도 디스크는 한 벌이고, 실제 바이트가
// 드는 것은 zip 을 만들 때뿐이다.
import { useState } from 'react'
import { Button, Card, Group, Stack, Text, ThemeIcon } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconDownload, IconPackageExport } from '@tabler/icons-react'
import { useMutation } from '@tanstack/react-query'
import {
  exportDataset,
  datasetExportUrl,
  type DatasetOut,
  type ExportKind,
  type ExportResult,
} from '../../api/client'

interface Props {
  projectId: string
  datasetId: string
  dataset: DatasetOut
}

const KINDS: { kind: ExportKind; label: string; hint: (d: DatasetOut) => string }[] = [
  {
    kind: 'train',
    label: 'Training data',
    hint: (d) => `train ${d.train} + val ${d.val}`,
  },
  { kind: 'test', label: 'Test data', hint: (d) => `test ${d.test}` },
  {
    kind: 'all',
    label: 'Everything',
    hint: (d) => `${d.reviewed} reviewed images`,
  },
]

function formatBytes(n: number): string {
  if (!n) return '–'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

export default function ExportCard({ projectId, datasetId, dataset }: Props) {
  const [ready, setReady] = useState<ExportResult | null>(null)

  const run = useMutation({
    mutationFn: (kind: ExportKind) => exportDataset(projectId, datasetId, kind),
    onSuccess: (r) => {
      setReady(r)
      notifications.show({ message: `Export ready: ${r.count} images`, color: 'green' })
    },
    onError: (e) => {
      setReady(null)
      notifications.show({ message: String(e), color: 'red' })
    },
  })

  const countFor = (kind: ExportKind) =>
    kind === 'train' ? dataset.train + dataset.val : kind === 'test' ? dataset.test : dataset.reviewed

  return (
    <Card withBorder radius="md" padding="lg">
      <Stack gap="md">
        <Group gap="xs">
          <ThemeIcon variant="light" size="lg" radius="md">
            <IconPackageExport size={20} />
          </ThemeIcon>
          <div>
            <Text fw={600}>Export</Text>
          </div>
        </Group>

        <Stack gap="xs">
          {KINDS.map((k) => (
            <Group key={k.kind} justify="space-between" wrap="nowrap">
              <div style={{ minWidth: 0 }}>
                <Text size="sm">{k.label}</Text>
                <Text size="xs" c="dimmed">
                  {k.hint(dataset)}
                </Text>
              </div>
              <Button
                variant="light"
                size="compact-sm"
                loading={run.isPending && run.variables === k.kind}
                disabled={countFor(k.kind) === 0}
                onClick={() => run.mutate(k.kind)}
              >
                Build
              </Button>
            </Group>
          ))}
        </Stack>

        {ready && (
          <Group justify="space-between" wrap="nowrap">
            <Text size="xs" c="dimmed">
              {ready.count} images · {ready.classes} classes · {formatBytes(ready.size_bytes)}
              {ready.copied === 0 ? ' · hard-linked' : ` · ${ready.copied} copied`}
            </Text>
            <Button
              component="a"
              href={datasetExportUrl(projectId, datasetId, ready.id)}
              leftSection={<IconDownload size={16} />}
            >
              Download
            </Button>
          </Group>
        )}
      </Stack>
    </Card>
  )
}

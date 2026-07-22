import { useState } from 'react'
import {
  Alert,
  Button,
  Group,
  Modal,
  NumberInput,
  SegmentedControl,
  Stack,
  Text,
} from '@mantine/core'
import { IconDownload } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  createExport,
  exportDownloadUrl,
  type ExportOut,
} from '../../api/client'

interface Props {
  projectId: string
  opened: boolean
  onClose: () => void
  /** target file names; null = all eligible images */
  names: string[] | null
}

export default function ExportModal({ projectId, opened, onClose, names }: Props) {
  const queryClient = useQueryClient()
  const [kind, setKind] = useState<'yolo' | 'images'>('yolo')
  const [valSplit, setValSplit] = useState<number | string>(0.2)
  const [seed, setSeed] = useState<number | string>(42)
  const [result, setResult] = useState<ExportOut | null>(null)

  const create = useMutation({
    mutationFn: () =>
      createExport(projectId, {
        kind,
        val_split: Number(valSplit),
        seed: Number(seed),
        names,
      }),
    onSuccess: (out) => {
      setResult(out)
      queryClient.invalidateQueries({ queryKey: ['exports', projectId] })
      queryClient.invalidateQueries({ queryKey: ['train-datasets'] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const close = () => {
    setResult(null)
    onClose()
  }

  return (
    <Modal opened={opened} onClose={close} title="Export" size="md">
      <Stack>
        <Text size="sm" c="dimmed">
          Target: {names ? `${names.length} selected images` : 'all images'}
        </Text>

        <SegmentedControl
          fullWidth
          value={kind}
          onChange={(v) => {
            setKind(v as 'yolo' | 'images')
            setResult(null)
          }}
          data={[
            { value: 'yolo', label: 'Label dataset (YOLO)' },
            { value: 'images', label: 'Original images only' },
          ]}
        />

        {kind === 'yolo' && (
          <Group grow>
            <NumberInput
              label="Val split"
              value={valSplit}
              onChange={setValSplit}
              min={0}
              max={0.9}
              step={0.05}
              decimalScale={2}
            />
            <NumberInput label="Shuffle seed" value={seed} onChange={setSeed} />
          </Group>
        )}
        {kind === 'yolo' && (
          <Text size="xs" c="dimmed">
            Only labeled images are included, with a train/val split and data.yaml.
          </Text>
        )}

        {result && (
          <Alert color="green" title="Export ready">
            <Stack gap="xs">
              <Text size="sm">
                {result.kind === 'yolo'
                  ? `train ${result.train} · val ${result.val} · ${result.classes} classes`
                  : `${result.count} images`}
                {' · '}
                {(result.size_bytes / 1024 / 1024).toFixed(1)} MB
              </Text>
              <Button
                component="a"
                href={exportDownloadUrl(projectId, result.id)}
                download
                leftSection={<IconDownload size={16} />}
                variant="light"
              >
                Download zip
              </Button>
            </Stack>
          </Alert>
        )}

        <Group justify="flex-end">
          <Button variant="default" onClick={close}>
            Close
          </Button>
          <Button onClick={() => create.mutate()} loading={create.isPending}>
            Create export
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}

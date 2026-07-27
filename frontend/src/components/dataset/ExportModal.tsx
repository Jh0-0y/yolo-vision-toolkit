import { useState } from 'react'
import {
  Button,
  Group,
  Modal,
  NumberInput,
  SegmentedControl,
  Stack,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation } from '@tanstack/react-query'
import { createExport } from '../../api/client'
import { useJobStore } from '../../stores/jobStore'

interface Props {
  projectId: string
  opened: boolean
  onClose: () => void
  /** target file names; null = all eligible images */
  names: string[] | null
}

export default function ExportModal({ projectId, opened, onClose, names }: Props) {
  const startExport = useJobStore((s) => s.trackExport)
  const [kind, setKind] = useState<'yolo' | 'images'>('yolo')
  const [valSplit, setValSplit] = useState<number | string>(0.2)
  const [seed, setSeed] = useState<number | string>(42)

  const create = useMutation({
    mutationFn: () =>
      createExport(projectId, {
        kind,
        val_split: Number(valSplit),
        seed: Number(seed),
        names,
      }),
    onSuccess: ({ export_id }) => {
      // hand off to the global job indicator; the finished export appears on the
      // Exports page where it can be downloaded.
      startExport(projectId, export_id, `Export · ${kind === 'yolo' ? 'YOLO' : 'images'}`)
      notifications.show({
        message: 'Export started — progress shows bottom-right; download it on the Exports page when done.',
        color: 'blue',
      })
      onClose()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  return (
    <Modal opened={opened} onClose={onClose} title="Export" size="md">
      <Stack>
        <Text size="sm" c="dimmed">
          Target: {names ? `${names.length} selected images` : 'all images'}
        </Text>

        <SegmentedControl
          fullWidth
          value={kind}
          onChange={(v) => setKind(v as 'yolo' | 'images')}
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

        <Group justify="flex-end">
          <Button variant="default" onClick={onClose}>
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

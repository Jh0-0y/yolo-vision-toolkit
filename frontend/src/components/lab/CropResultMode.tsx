import { useEffect, useState } from 'react'
import { Alert, Anchor, Card, Stack, Text } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconFileCode, IconX } from '@tabler/icons-react'
import { Link } from 'react-router-dom'
import type { DetectorPayload, ModelOut, TrackcropOverrides } from '../../api/client'
import DetectionSettings, { newEntry, type DetectorEntry } from './DetectionSettings'
import TuningPanel from './TuningPanel'
import { useAnnotateJob } from './useAnnotateJob'

function entryPayload(entries: DetectorEntry[]): DetectorPayload[] {
  return entries.filter((e) => e.modelId).map((e) => ({
    model_id: e.modelId as string,
    mode: e.mode,
    conf: e.conf === '' ? undefined : e.conf,
    imgsz: e.imgsz,
    tile_size: e.tileSize,
    stride: e.stride,
    merge_iou: e.mergeIou,
  }))
}

const VIDEO_MIME = [
  'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo',
]

interface Props {
  projectId: string
  models: ModelOut[]
}

/** Crop Result — produces crop.json coordinates only (no video render). For inspecting coords after tuning. */
export default function CropResultMode({ projectId, models }: Props) {
  const [entries, setEntries] = useState<DetectorEntry[]>(() => [newEntry('full')])
  const [sampling, setSampling] = useState<number | ''>('')
  const base = entries[0]
  const [overrides, setOverrides] = useState<TrackcropOverrides>({})
  const job = useAnnotateJob(projectId)

  useEffect(() => {
    if (models.length && !entries[0].modelId)
      setEntries((prev) => [{ ...prev[0], modelId: models[0].id }, ...prev.slice(1)])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file || !base.modelId) return
    await job.run({
      file,
      modelIds: [base.modelId],
      projectId,
      params: {
        conf: base.conf === '' ? undefined : base.conf,
        iou_wbf: 0.7,
        imgsz: base.imgsz,
        device: null,
      },
      objectTracking: false,
      cropTracking: true,
      cropOutput: 'none',
      overrides: sampling === '' ? overrides : { ...overrides, sampling_interval_ms: sampling },
      detectors: entryPayload(entries),
    })
  }

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <DetectionSettings
            models={models}
            entries={entries}
            onEntries={setEntries}
            sampling={sampling}
            onSampling={setSampling}
            disabled={job.starting}
          />
          <TuningPanel
            value={overrides}
            onChange={setOverrides}
            disabled={job.starting}
            exclude={['sampling_interval_ms']}
          />
        </Stack>
      </Card>

      <Card withBorder radius="md" padding="md">
        <Stack gap="md">
          {job.error && (
            <Alert color="red" icon={<IconAlertTriangle size={18} />} withCloseButton onClose={() => job.setError(null)}>
              {job.error}
            </Alert>
          )}

          <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} disabled={!base.modelId || job.starting}>
            <Stack align="center" gap="xs" py="xl">
              <Dropzone.Idle><IconFileCode size={40} stroke={1.2} /></Dropzone.Idle>
              <Dropzone.Reject><IconX size={40} /></Dropzone.Reject>
              <Text size="sm">Drop a video (mp4, mov, …) or click to upload</Text>
              <Text size="xs" c="dimmed">
                Computes crop coordinates (crop.json) only — no video, so it's fast.
              </Text>
            </Stack>
          </Dropzone>

          {job.startedName && (
            <Text size="sm" c="dimmed">
              Started <b>{job.startedName}</b> — it keeps running if you leave this tab.{' '}
              <Anchor component={Link} to={`/projects/${projectId}/lab/crops`}>
                Crop Runs
              </Anchor>{' '}
              has the progress and the result.
            </Text>
          )}
        </Stack>
      </Card>
    </Stack>
  )
}

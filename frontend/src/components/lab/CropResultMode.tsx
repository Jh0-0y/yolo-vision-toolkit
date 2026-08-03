import { useEffect, useState } from 'react'
import {
  Alert,
  Anchor,
  Button,
  Card,
  Code,
  Group,
  Progress,
  Stack,
  Text,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconFileCode, IconX } from '@tabler/icons-react'
import { annotateCropUrl, type DetectorPayload, type ModelOut, type TrackcropOverrides } from '../../api/client'
import DetectionSettings, { newEntry, type DetectorEntry } from './DetectionSettings'
import TuningPanel from './TuningPanel'
import { PHASE_LABEL, useAnnotateJob } from './useAnnotateJob'

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
  const [summary, setSummary] = useState<Record<string, number> | null>(null)
  const job = useAnnotateJob()

  useEffect(() => {
    if (models.length && !entries[0].modelId)
      setEntries((prev) => [{ ...prev[0], modelId: models[0].id }, ...prev.slice(1)])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

  // on done, fetch crop.json to show its summary (coverage)
  useEffect(() => {
    if (!job.done || !job.jobId) return
    let alive = true
    fetch(annotateCropUrl(job.jobId))
      .then((r) => r.json())
      .then((d) => alive && setSummary(d.summary ?? null))
      .catch(() => alive && setSummary(null))
    return () => {
      alive = false
    }
  }, [job.done, job.jobId])

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file || !base.modelId) return
    setSummary(null)
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
            disabled={job.running}
          />
          <TuningPanel
            value={overrides}
            onChange={setOverrides}
            disabled={job.running}
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

          {!job.fileName ? (
            <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} disabled={!base.modelId}>
              <Stack align="center" gap="xs" py="xl">
                <Dropzone.Idle><IconFileCode size={40} stroke={1.2} /></Dropzone.Idle>
                <Dropzone.Reject><IconX size={40} /></Dropzone.Reject>
                <Text size="sm">Drop a video (mp4, mov, …) or click to upload</Text>
                <Text size="xs" c="dimmed">
                  Computes crop coordinates (crop.json) only — no video, so it's fast.
                </Text>
              </Stack>
            </Dropzone>
          ) : (
            <Stack gap="sm">
              <Group justify="space-between">
                <Text size="sm" truncate="end" maw={360}>{job.fileName}</Text>
                <Button size="xs" variant="subtle" onClick={() => { job.reset(); setSummary(null) }} disabled={job.running}>
                  New video
                </Button>
              </Group>

              {job.running && (
                <Stack gap={4}>
                  <Text size="sm">{PHASE_LABEL[job.progress?.phase ?? 'start'] ?? 'Working…'}</Text>
                  <Progress value={job.progress?.phase === 'crop_analyze' ? 60 : 30} animated />
                </Stack>
              )}

              {job.done && job.jobId && (
                <Stack gap="xs">
                  {summary && (
                    <Card withBorder radius="sm" padding="sm" bg="var(--mantine-color-default-hover)">
                      <Text size="sm" fw={600} mb={4}>Coverage</Text>
                      <Group gap="lg">
                        <Text size="sm">Ball tracking <Code>{pct(summary.ball_tracking_coverage)}</Code></Text>
                        <Text size="sm">Player tracking <Code>{pct(summary.player_tracking_coverage)}</Code></Text>
                        <Text size="sm">Center fallback <Code>{pct(summary.fallback_coverage)}</Code></Text>
                        <Text size="sm">Keyframes <Code>{summary.keyframe_count}</Code></Text>
                      </Group>
                    </Card>
                  )}
                  <Group>
                    <Anchor href={annotateCropUrl(job.jobId)} download={`crop_${job.fileName}.json`} size="sm">
                      Download crop coordinates (JSON)
                    </Anchor>
                    <Text c="dimmed" size="xs">Stored temporarily; auto-deleted after 1 hour.</Text>
                  </Group>
                </Stack>
              )}
            </Stack>
          )}
        </Stack>
      </Card>
    </Stack>
  )
}

function pct(v: number | undefined): string {
  return v == null ? '—' : `${Math.round(v * 100)}%`
}

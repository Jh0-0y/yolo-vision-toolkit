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
import { annotateCropUrl, type ModelOut, type TrackcropOverrides } from '../../api/client'
import DetectionSettings from './DetectionSettings'
import TuningPanel from './TuningPanel'
import { DEFAULT_IMGSZ, PHASE_LABEL, useAnnotateJob } from './useAnnotateJob'

const VIDEO_MIME = [
  'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo',
]

interface Props {
  projectId: string
  models: ModelOut[]
}

/** Crop Result — produces crop.json coordinates only (no video render). For inspecting coords after tuning. */
export default function CropResultMode({ projectId, models }: Props) {
  const [modelId, setModelId] = useState<string | null>(null)
  const [conf, setConf] = useState<number | ''>('')
  const [imgsz, setImgsz] = useState(DEFAULT_IMGSZ)
  const [sampling, setSampling] = useState<number | ''>('')
  const [overrides, setOverrides] = useState<TrackcropOverrides>({})
  const [summary, setSummary] = useState<Record<string, number> | null>(null)
  const job = useAnnotateJob()

  useEffect(() => {
    if (models.length && !modelId) setModelId(models[0].id)
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
    if (!file || !modelId) return
    setSummary(null)
    await job.run({
      file,
      modelIds: [modelId],
      projectId,
      params: { conf: conf === '' ? undefined : conf, iou_wbf: 0.7, imgsz, device: null },
      objectTracking: false,
      cropTracking: true,
      cropOutput: 'none',
      overrides: sampling === '' ? overrides : { ...overrides, sampling_interval_ms: sampling },
    })
  }

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <DetectionSettings
            models={models}
            modelId={modelId}
            onModelId={setModelId}
            imgsz={imgsz}
            onImgsz={setImgsz}
            conf={conf}
            onConf={setConf}
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
            <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} disabled={!modelId}>
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

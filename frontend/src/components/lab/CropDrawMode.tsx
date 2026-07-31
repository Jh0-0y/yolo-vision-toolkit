import { useEffect, useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Checkbox,
  Group,
  NumberInput,
  Progress,
  Select,
  Stack,
  Text,
  UnstyledButton,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconChevronRight, IconMovie, IconX } from '@tabler/icons-react'
import {
  annotateCropUrl,
  annotateResultUrl,
  type ModelOut,
  type TrackcropOverrides,
} from '../../api/client'
import TuningPanel from './TuningPanel'
import { DEFAULT_IMGSZ, IMGSZ_OPTIONS, PHASE_LABEL, useAnnotateJob } from './useAnnotateJob'

const VIDEO_MIME = [
  'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo',
]

interface Props {
  projectId: string
  models: ModelOut[]
}

/** Crop Draw Tool — renders overlays (objects / crop box / dead zone / highlight) onto the full video. */
export default function CropDrawMode({ projectId, models }: Props) {
  const [modelId, setModelId] = useState<string | null>(null)
  const [conf, setConf] = useState<number | ''>('')
  const [imgsz, setImgsz] = useState(DEFAULT_IMGSZ)
  const [objInference, setObjInference] = useState(true)
  const [drawCropBox, setDrawCropBox] = useState(true)
  const [showDeadZone, setShowDeadZone] = useState(true)
  const [showCenterLine, setShowCenterLine] = useState(true)
  const [showHighlight, setShowHighlight] = useState(true)
  const [showOverlays, setShowOverlays] = useState(true)
  const [overrides, setOverrides] = useState<TrackcropOverrides>({})
  const [videoFailed, setVideoFailed] = useState(false)
  const [ranCrop, setRanCrop] = useState(false)
  const job = useAnnotateJob()

  useEffect(() => {
    if (models.length && !modelId) setModelId(models[0].id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

  // crop analysis is needed only when the box (rect + dead zone/center line) or highlight is on
  const cropTracking = drawCropBox || showHighlight
  const nothingSelected = !objInference && !drawCropBox && !showHighlight

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file || !modelId || nothingSelected) return
    setVideoFailed(false)
    setRanCrop(cropTracking)
    await job.run({
      file,
      modelIds: [modelId],
      projectId,
      params: { conf: conf === '' ? undefined : conf, iou_wbf: 0.7, imgsz, device: null },
      objectTracking: objInference,
      cropTracking,
      cropOutput: 'label',
      drawCropBox,
      showDeadZone,
      showCenterLine,
      showTargetHighlight: showHighlight,
      overrides,
    })
  }

  function reset() {
    job.reset()
    setVideoFailed(false)
    setRanCrop(false)
  }

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <Text size="sm" fw={600}>Inference</Text>
          <Select
            label="Model"
            placeholder={models.length ? 'Pick a model' : 'No models — train or upload one first'}
            data={models.map((m) => ({ value: m.id, label: m.name }))}
            value={modelId}
            onChange={setModelId}
            disabled={job.running || !models.length}
            allowDeselect={false}
          />
          <Group grow align="flex-start">
            <Select
              label="Image size (inference)"
              description="Match your model's training size"
              data={IMGSZ_OPTIONS}
              value={String(imgsz)}
              onChange={(v) => v && setImgsz(Number(v))}
              disabled={job.running}
              allowDeselect={false}
            />
            <NumberInput
              label="Confidence"
              description="Detection threshold"
              placeholder="default 0.1"
              value={conf}
              onChange={(v) => setConf(v === '' || v == null ? '' : Number(v))}
              min={0.05}
              max={0.95}
              step={0.05}
              decimalScale={2}
              disabled={job.running}
            />
          </Group>

          <TuningPanel value={overrides} onChange={setOverrides} disabled={job.running} />

          <Stack gap={4}>
            <UnstyledButton onClick={() => setShowOverlays((v) => !v)} disabled={job.running}>
              <Group gap={4}>
                <IconChevronRight
                  size={14}
                  style={{
                    transform: showOverlays ? 'rotate(90deg)' : 'none',
                    transition: 'transform 150ms',
                  }}
                />
                <Text size="sm" fw={600}>Overlays</Text>
              </Group>
            </UnstyledButton>
            {showOverlays && (
              <Stack gap={4} pl="lg">
                <Checkbox
                  label="Object inference — ByteTrack boxes, IDs & conf"
                  checked={objInference}
                  onChange={(e) => setObjInference(e.currentTarget.checked)}
                  disabled={job.running}
                  size="xs"
                />
                <Checkbox
                  label="Crop tracking box — 9:16 rectangle"
                  checked={drawCropBox}
                  onChange={(e) => setDrawCropBox(e.currentTarget.checked)}
                  disabled={job.running}
                  size="xs"
                />
                {drawCropBox && (
                  <Stack gap={4} pl="lg">
                    <Checkbox
                      label="Dead zone band"
                      checked={showDeadZone}
                      onChange={(e) => setShowDeadZone(e.currentTarget.checked)}
                      disabled={job.running}
                      size="xs"
                    />
                    <Checkbox
                      label="Center line & type label"
                      checked={showCenterLine}
                      onChange={(e) => setShowCenterLine(e.currentTarget.checked)}
                      disabled={job.running}
                      size="xs"
                    />
                  </Stack>
                )}
                <Checkbox
                  label="Target highlight — selected ball (▼) & carrier markers"
                  checked={showHighlight}
                  onChange={(e) => setShowHighlight(e.currentTarget.checked)}
                  disabled={job.running}
                  size="xs"
                />
                {nothingSelected && (
                  <Text size="xs" c="red">Select at least one overlay to draw.</Text>
                )}
              </Stack>
            )}
          </Stack>
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
            <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} disabled={!modelId || nothingSelected}>
              <Stack align="center" gap="xs" py="xl">
                <Dropzone.Idle><IconMovie size={40} stroke={1.2} /></Dropzone.Idle>
                <Dropzone.Reject><IconX size={40} /></Dropzone.Reject>
                <Text size="sm">Drop a video (mp4, mov, …) or click to upload</Text>
                <Text size="xs" c="dimmed">Returns a video with the overlays you selected.</Text>
              </Stack>
            </Dropzone>
          ) : (
            <Stack gap="sm">
              <Group justify="space-between">
                <Text size="sm" truncate="end" maw={360}>{job.fileName}</Text>
                <Button size="xs" variant="subtle" onClick={reset} disabled={job.running}>New video</Button>
              </Group>

              {job.running && (
                <Stack gap={4}>
                  <Group justify="space-between">
                    <Text size="sm">{PHASE_LABEL[job.progress?.phase ?? 'start'] ?? 'Working…'}</Text>
                    {job.progress?.phase === 'annotate' && <Badge variant="light">{job.pct}%</Badge>}
                  </Group>
                  <Progress value={job.progress?.phase === 'encoding' ? 100 : job.pct} animated />
                </Stack>
              )}

              {job.done && job.jobId && (
                <Stack gap="xs">
                  {videoFailed ? (
                    <Alert color="orange" icon={<IconAlertTriangle size={18} />}>
                      The video couldn't play inline (unsupported codec in this browser). Download it to view.
                    </Alert>
                  ) : (
                    // eslint-disable-next-line jsx-a11y/media-has-caption
                    <video
                      src={annotateResultUrl(job.jobId)}
                      controls
                      onError={() => setVideoFailed(true)}
                      style={{ display: 'block', margin: '0 auto', maxWidth: '100%', maxHeight: '70vh', borderRadius: 8 }}
                    />
                  )}
                  <Group>
                    <Anchor href={annotateResultUrl(job.jobId)} download={`tracked_${job.fileName}`} size="sm">
                      Download result
                    </Anchor>
                    {ranCrop && (
                      <Anchor href={annotateCropUrl(job.jobId)} download={`crop_${job.fileName}.json`} size="sm">
                        Download crop coordinates (JSON)
                      </Anchor>
                    )}
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

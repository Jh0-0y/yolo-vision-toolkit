import { useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  FileButton,
  Group,
  Progress,
  Radio,
  Stack,
  Text,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconFileText, IconMovie, IconX } from '@tabler/icons-react'
import { annotateResultUrl, startCropCut } from '../../api/client'
import { PHASE_LABEL, useAnnotateJob } from './useAnnotateJob'

const VIDEO_MIME = [
  'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo',
]

type CutMode = 'center' | 'json'

/** Crop Video — cut a vertical 9:16 clip with NO inference. Two options:
 *  - Center crop: fixed to the frame centre (no JSON).
 *  - JSON to crop: follow an uploaded crop.json trajectory. */
export default function CropVideoMode() {
  const [mode, setMode] = useState<CutMode>('json')
  const [cropJson, setCropJson] = useState<File | null>(null)
  const [videoFailed, setVideoFailed] = useState(false)
  const job = useAnnotateJob()

  const needsJson = mode === 'json'
  const blocked = needsJson && !cropJson

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file || blocked) return
    setVideoFailed(false)
    await job.launch(file.name, () =>
      startCropCut({ file, mode, cropJson: needsJson ? cropJson ?? undefined : undefined }),
    )
  }

  function reset() {
    job.reset()
    setVideoFailed(false)
  }

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <Text size="sm" fw={600}>Mode</Text>
          <Radio.Group value={mode} onChange={(v) => setMode(v as CutMode)}>
            <Stack gap={6}>
              <Radio
                value="center"
                label="Center crop — fixed to the frame centre (no JSON)"
                disabled={job.running}
              />
              <Radio
                value="json"
                label="JSON to crop — follow an uploaded crop.json"
                disabled={job.running}
              />
            </Stack>
          </Radio.Group>

          {needsJson && (
            <Group gap="sm">
              <FileButton onChange={setCropJson} accept="application/json,.json">
                {(props) => (
                  <Button
                    {...props}
                    size="xs"
                    variant="light"
                    leftSection={<IconFileText size={16} />}
                    disabled={job.running}
                  >
                    {cropJson ? 'Change crop.json' : 'Choose crop.json'}
                  </Button>
                )}
              </FileButton>
              {cropJson ? (
                <Text size="xs" truncate="end" maw={280}>{cropJson.name}</Text>
              ) : (
                <Text size="xs" c="dimmed">Pick the crop.json to follow.</Text>
              )}
            </Group>
          )}
          <Text size="xs" c="dimmed">No model — just cuts the vertical 9:16 clip.</Text>
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
            <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} disabled={blocked}>
              <Stack align="center" gap="xs" py="xl">
                <Dropzone.Idle><IconMovie size={40} stroke={1.2} /></Dropzone.Idle>
                <Dropzone.Reject><IconX size={40} /></Dropzone.Reject>
                <Text size="sm">Drop a video (mp4, mov, …) or click to upload</Text>
                <Text size="xs" c="dimmed">
                  {needsJson
                    ? 'Cuts the clip following the uploaded crop.json.'
                    : 'Cuts a centre-fixed vertical clip.'}
                </Text>
                {blocked && <Text size="xs" c="red">Choose a crop.json first.</Text>}
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
                    <Anchor href={annotateResultUrl(job.jobId)} download={`crop_${job.fileName}`} size="sm">
                      Download cropped video
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

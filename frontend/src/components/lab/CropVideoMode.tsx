import { useState } from 'react'
import {
  Alert,
  Anchor,
  Button,
  Card,
  FileButton,
  Group,
  Radio,
  Stack,
  Text,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconFileText, IconMovie, IconX } from '@tabler/icons-react'
import { Link } from 'react-router-dom'
import { startCropCut } from '../../api/client'
import { useAnnotateJob } from './useAnnotateJob'

const VIDEO_MIME = [
  'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo',
]

type CutMode = 'center' | 'json'

interface Props {
  projectId: string
}

/** Crop Video — cut a vertical 9:16 clip with NO inference. Two options:
 *  - Center crop: fixed to the frame centre (no JSON).
 *  - JSON to crop: follow an uploaded crop.json trajectory. */
export default function CropVideoMode({ projectId }: Props) {
  const [mode, setMode] = useState<CutMode>('json')
  const [cropJson, setCropJson] = useState<File | null>(null)
  const job = useAnnotateJob(projectId)

  const needsJson = mode === 'json'
  const blocked = needsJson && !cropJson

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file || blocked) return
    await job.launch(file.name, () =>
      startCropCut({
        file,
        projectId,
        mode,
        cropJson: needsJson ? cropJson ?? undefined : undefined,
      }),
    )
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
                disabled={job.starting}
              />
              <Radio
                value="json"
                label="JSON to crop — follow an uploaded crop.json"
                disabled={job.starting}
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
                    disabled={job.starting}
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

          <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} disabled={blocked || job.starting}>
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

          {job.startedName && (
            <Text size="sm" c="dimmed">
              Started <b>{job.startedName}</b> — it keeps running if you leave this tab.{' '}
              <Anchor component={Link} to={`/projects/${projectId}/lab/crops`}>
                Crop Runs
              </Anchor>{' '}
              has the progress and the clip.
            </Text>
          )}
        </Stack>
      </Card>
    </Stack>
  )
}

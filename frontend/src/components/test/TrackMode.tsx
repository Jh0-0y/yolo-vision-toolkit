import { useEffect, useRef, useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Group,
  Progress,
  Select,
  Slider,
  Stack,
  Text,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconMovie, IconX } from '@tabler/icons-react'
import {
  annotateResultUrl,
  startAnnotate,
  subscribeAnnotateEvents,
  type AnnotateProgress,
  type ModelOut,
} from '../../api/client'

const VIDEO_MIME = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo']
const IMGSZ = ['320', '480', '640', '960', '1280']

interface Props {
  projectId: string
  models: ModelOut[]
}

const PHASE_LABEL: Record<string, string> = {
  start: 'Preparing…',
  annotate: 'Tracking objects…',
  encoding: 'Encoding video…',
  done: 'Done',
}

/** Run a model's object tracker (ByteTrack) over a video and play back the
 *  annotated result (persistent per-object IDs + motion trails). */
export default function TrackMode({ projectId, models }: Props) {
  const [modelId, setModelId] = useState<string | null>(null)
  const [conf, setConf] = useState(0.4)
  const [imgsz, setImgsz] = useState('640')
  const [fileName, setFileName] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<AnnotateProgress | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const [videoFailed, setVideoFailed] = useState(false)
  const unsub = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (models.length && !modelId) setModelId(models[0].id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])
  useEffect(() => () => unsub.current?.(), [])

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file || !modelId) return
    unsub.current?.()
    setFileName(file.name)
    setError(null)
    setDone(false)
    setVideoFailed(false)
    setJobId(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      const { job_id } = await startAnnotate({
        file,
        modelIds: [modelId],
        projectId,
        params: { conf, iou_wbf: 0.7, imgsz: Number(imgsz), device: null },
      })
      setJobId(job_id)
      unsub.current = subscribeAnnotateEvents(job_id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          setDone(true)
          setRunning(false)
        } else if (ev.phase === 'error') {
          setError(ev.msg || 'Tracking failed')
          setRunning(false)
        } else if (ev.phase === 'cancelled') {
          setRunning(false)
        }
      })
    } catch (e) {
      setError((e as Error).message)
      setRunning(false)
    }
  }

  function reset() {
    unsub.current?.()
    setFileName(null)
    setJobId(null)
    setProgress(null)
    setRunning(false)
    setError(null)
    setDone(false)
    setVideoFailed(false)
  }

  const pct =
    progress?.total && progress.done != null ? Math.round((progress.done / progress.total) * 100) : 0

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <Text size="sm" fw={600}>
            Settings
          </Text>
          <Select
            label="Model"
            placeholder={models.length ? 'Pick a model' : 'No models — train or upload one first'}
            data={models.map((m) => ({ value: m.id, label: m.name }))}
            value={modelId}
            onChange={setModelId}
            disabled={running || !models.length}
            allowDeselect={false}
          />
          <Group grow>
            <div>
              <Text size="sm" fw={600}>
                Confidence <Text span c="dimmed">{conf.toFixed(2)}</Text>
              </Text>
              <Slider min={0.05} max={0.95} step={0.05} value={conf} onChange={setConf} disabled={running} />
            </div>
            <Select
              label="Image size"
              data={IMGSZ}
              value={imgsz}
              onChange={(v) => v && setImgsz(v)}
              allowDeselect={false}
              disabled={running}
            />
          </Group>
        </Stack>
      </Card>

      <Card withBorder radius="md" padding="md">
        <Stack gap="md">
          {error && (
            <Alert color="red" icon={<IconAlertTriangle size={18} />} withCloseButton onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {!fileName ? (
            <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} disabled={!modelId}>
              <Stack align="center" gap="xs" py="xl">
                <Dropzone.Idle>
                  <IconMovie size={40} stroke={1.2} />
                </Dropzone.Idle>
                <Dropzone.Reject>
                  <IconX size={40} />
                </Dropzone.Reject>
                <Text size="sm">Drop a video (mp4, mov, …) or click to upload</Text>
                <Text size="xs" c="dimmed">
                  The model tracks objects across frames and returns an annotated video with IDs.
                </Text>
              </Stack>
            </Dropzone>
          ) : (
            <Stack gap="sm">
              <Group justify="space-between">
                <Text size="sm" truncate="end" maw={360}>
                  {fileName}
                </Text>
                <Button size="xs" variant="subtle" onClick={reset} disabled={running}>
                  New video
                </Button>
              </Group>

              {running && (
                <Stack gap={4}>
                  <Group justify="space-between">
                    <Text size="sm">
                      {PHASE_LABEL[progress?.phase ?? 'start'] ?? 'Working…'}
                      {progress?.phase === 'annotate' && ` ${progress?.done ?? 0}/${progress?.total ?? '?'}`}
                    </Text>
                    {progress?.phase === 'annotate' && <Badge variant="light">{pct}%</Badge>}
                  </Group>
                  <Progress value={progress?.phase === 'encoding' ? 100 : pct} animated />
                </Stack>
              )}

              {done && jobId && (
                <Stack gap="xs">
                  {videoFailed ? (
                    <Alert color="orange" icon={<IconAlertTriangle size={18} />}>
                      The video couldn't play inline (unsupported codec in this browser). Download it
                      to view.
                    </Alert>
                  ) : (
                    // eslint-disable-next-line jsx-a11y/media-has-caption
                    <video
                      src={annotateResultUrl(jobId)}
                      controls
                      onError={() => setVideoFailed(true)}
                      style={{ width: '100%', borderRadius: 8 }}
                    />
                  )}
                  <Group>
                    <Anchor href={annotateResultUrl(jobId)} download={`tracked_${fileName}`} size="sm">
                      Download result
                    </Anchor>
                    <Text c="dimmed" size="xs">
                      Stored temporarily; auto-deleted after 1 hour.
                    </Text>
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

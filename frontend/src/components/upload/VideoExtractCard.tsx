import { useState } from 'react'
import {
  Badge,
  Button,
  Card,
  Collapse,
  Group,
  NumberInput,
  Progress,
  Slider,
  Stack,
  Switch,
  Text,
  ThemeIcon,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconMovie, IconSettings, IconX } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useQueryClient } from '@tanstack/react-query'
import {
  subscribeVideoEvents,
  uploadVideo,
  type VideoProgressEvent,
} from '../../api/client'

const VIDEO_MIME = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm']

export default function VideoExtractCard({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient()
  const [file, setFile] = useState<File | null>(null)
  const [targetFps, setTargetFps] = useState<number>(2)
  const [maxFrames, setMaxFrames] = useState<number>(2000)
  const [startSec, setStartSec] = useState<number>(0)
  const [endSec, setEndSec] = useState<number | ''>('')
  const [dedup, setDedup] = useState(true)
  const [dedupThreshold, setDedupThreshold] = useState(0.92)
  const [advanced, setAdvanced] = useState(false)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<VideoProgressEvent | null>(null)

  const start = async () => {
    if (!file) return
    setBusy(true)
    setProgress(null)
    try {
      const { video_id } = await uploadVideo(projectId, file, {
        target_fps: targetFps,
        max_frames: maxFrames,
        start_sec: startSec,
        end_sec: endSec === '' ? null : Number(endSec),
        dedup,
        dedup_threshold: dedupThreshold,
      })
      subscribeVideoEvents(projectId, video_id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          notifications.show({
            message: `${ev.saved} frames extracted${ev.skipped_dup ? `, ${ev.skipped_dup} duplicates skipped` : ''}`,
            color: 'green',
          })
          queryClient.invalidateQueries({ queryKey: ['images', projectId] })
          queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
          setBusy(false)
          setFile(null)
        } else if (ev.phase === 'error' || ev.phase === 'cancelled') {
          notifications.show({ message: ev.msg || 'Extraction stopped', color: 'red' })
          setBusy(false)
        }
      })
    } catch (e) {
      notifications.show({ message: String(e), color: 'red' })
      setBusy(false)
    }
  }

  const pct =
    progress?.total_frames && progress.scanned
      ? Math.min(100, Math.round((progress.scanned / progress.total_frames) * 100))
      : progress?.phase === 'done'
        ? 100
        : 0

  return (
    <Card withBorder radius="md" padding="lg">
      <Stack>
        <Group gap="xs">
          <ThemeIcon variant="light" size="lg" radius="md">
            <IconMovie size={20} />
          </ThemeIcon>
          <div>
            <Text fw={600}>Video upload</Text>
            <Text size="xs" c="dimmed">
              Extract frames from a video into the dataset
            </Text>
          </div>
        </Group>

        {!file ? (
          <Dropzone
            onDrop={(files) => setFile(files[0] ?? null)}
            accept={VIDEO_MIME}
            multiple={false}
            disabled={busy}
            radius="md"
          >
            <Stack align="center" gap={6} py="lg" style={{ pointerEvents: 'none' }}>
              <IconMovie size={32} stroke={1.4} />
              <Text size="sm">Drag a video here or click to browse</Text>
              <Text size="xs" c="dimmed">
                mp4 · mov · avi · mkv · webm
              </Text>
            </Stack>
          </Dropzone>
        ) : (
          <Group justify="space-between" wrap="nowrap">
            <Badge variant="light" size="lg" style={{ maxWidth: '80%' }}>
              <Text size="sm" truncate>
                {file.name}
              </Text>
            </Badge>
            <Button
              variant="subtle"
              color="gray"
              size="compact-sm"
              leftSection={<IconX size={14} />}
              onClick={() => setFile(null)}
              disabled={busy}
            >
              Remove
            </Button>
          </Group>
        )}

        <Group grow>
          <NumberInput
            label="Frames per second"
            description="Independent of the source fps"
            value={targetFps}
            onChange={(v) => setTargetFps(Number(v) || 1)}
            min={0.1}
            step={0.5}
            disabled={busy}
          />
          <NumberInput
            label="Max frames"
            value={maxFrames}
            onChange={(v) => setMaxFrames(Number(v) || 1)}
            min={1}
            disabled={busy}
          />
        </Group>

        <Switch
          label="Skip near-duplicate frames (dedup)"
          description="Skips adjacent, nearly identical frames"
          checked={dedup}
          onChange={(e) => setDedup(e.currentTarget.checked)}
          disabled={busy}
        />

        <Button
          variant="subtle"
          size="compact-sm"
          leftSection={<IconSettings size={14} />}
          onClick={() => setAdvanced((v) => !v)}
          style={{ alignSelf: 'flex-start' }}
        >
          Advanced
        </Button>
        <Collapse expanded={advanced}>
          <Stack gap="sm">
            <Group grow>
              <NumberInput
                label="Start (sec)"
                value={startSec}
                onChange={(v) => setStartSec(Number(v) || 0)}
                min={0}
                disabled={busy}
              />
              <NumberInput
                label="End (sec)"
                description="Empty = until the end"
                value={endSec}
                onChange={(v) => setEndSec(v === '' ? '' : Number(v))}
                min={0}
                disabled={busy}
              />
            </Group>
            {dedup && (
              <div>
                <Text size="sm" mb={4}>
                  Dedup threshold: {dedupThreshold.toFixed(2)}
                </Text>
                <Slider
                  value={dedupThreshold}
                  onChange={setDedupThreshold}
                  min={0.8}
                  max={1}
                  step={0.01}
                  disabled={busy}
                  label={(v) => v.toFixed(2)}
                />
                <Text size="xs" c="dimmed" mt={4}>
                  Higher = frames must be more similar to be skipped
                </Text>
              </div>
            )}
          </Stack>
        </Collapse>

        {progress && (
          <Stack gap={4}>
            <Progress value={pct} animated={busy} />
            <Text size="xs" c="dimmed">
              {progress.phase === 'done'
                ? `Done — ${progress.saved} frames extracted`
                : progress.phase === 'start'
                  ? `Analyzing… (source ${progress.src_fps}fps, ${progress.total_frames} frames)`
                  : `Extracting… ${progress.saved ?? 0} saved (scanned ${progress.scanned ?? 0}/${progress.total_frames ?? '?'})`}
            </Text>
          </Stack>
        )}

        <Button onClick={start} disabled={!file || busy} loading={busy}>
          Start extraction
        </Button>
      </Stack>
    </Card>
  )
}

import { useState } from 'react'
import {
  Badge,
  Button,
  Card,
  Collapse,
  Group,
  Input,
  NumberInput,
  Slider,
  Stack,
  Switch,
  Text,
  ThemeIcon,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconMovie, IconSettings, IconX } from '@tabler/icons-react'
import { useJobStore } from '../../stores/jobStore'
import TilingOptions, { DEFAULT_TILING, type TilingState } from './TilingOptions'

const VIDEO_MIME = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm']

export default function VideoExtractCard({ projectId }: { projectId: string }) {
  const startVideoJob = useJobStore((s) => s.startVideoJob)
  // one extraction at a time (the server runs a single video worker)
  const videoBusy = useJobStore((s) =>
    Object.values(s.jobs).some((j) => j.kind === 'video' && j.status === 'running'),
  )
  const [file, setFile] = useState<File | null>(null)
  const [targetFps, setTargetFps] = useState<number>(2)
  const [maxFrames, setMaxFrames] = useState<number>(2000)
  const [startSec, setStartSec] = useState<number>(0)
  const [endSec, setEndSec] = useState<number | ''>('')
  const [dedup, setDedup] = useState(true)
  const [dedupThreshold, setDedupThreshold] = useState(0.92)
  const [tiling, setTiling] = useState<TilingState>(DEFAULT_TILING)
  const [advanced, setAdvanced] = useState(false)

  const start = () => {
    if (!file) return
    // hand off to the global job store — upload % + server extraction progress
    // are shown by the app-wide JobIndicator and survive navigation.
    startVideoJob(projectId, file, {
      target_fps: targetFps,
      max_frames: maxFrames,
      start_sec: startSec,
      end_sec: endSec === '' ? null : Number(endSec),
      dedup,
      dedup_threshold: dedupThreshold,
      tile: tiling.tile,
      tile_size: tiling.tileSize,
      stride: tiling.stride,
    })
    setFile(null)
  }

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
            disabled={videoBusy}
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
            >
              Remove
            </Button>
          </Group>
        )}

        <Group grow align="flex-start">
          <NumberInput
            label="Frames per second"
            description="Independent of the source fps"
            value={targetFps}
            onChange={(v) => setTargetFps(Number(v) || 1)}
            min={0.1}
            step={0.5}
            disabled={videoBusy}
          />
          <NumberInput
            label="Max frames"
            description="Cap on total saved frames"
            value={maxFrames}
            onChange={(v) => setMaxFrames(Number(v) || 1)}
            min={1}
            disabled={videoBusy}
          />
        </Group>

        <Switch
          label="Skip near-duplicate frames (dedup)"
          description="Skips adjacent, nearly identical frames"
          checked={dedup}
          onChange={(e) => setDedup(e.currentTarget.checked)}
          disabled={videoBusy}
        />

        <TilingOptions value={tiling} onChange={setTiling} disabled={videoBusy} />

        <Stack gap="sm">
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
              <Group grow align="flex-start">
                <NumberInput
                  label="Start (sec)"
                  description="0 = from the beginning"
                  value={startSec}
                  onChange={(v) => setStartSec(Number(v) || 0)}
                  min={0}
                  disabled={videoBusy}
                />
                <NumberInput
                  label="End (sec)"
                  description="Empty = until the end"
                  value={endSec}
                  onChange={(v) => setEndSec(v === '' ? '' : Number(v))}
                  min={0}
                  disabled={videoBusy}
                />
              </Group>
              {dedup && (
                <Input.Wrapper
                  label={`Dedup threshold: ${dedupThreshold.toFixed(2)}`}
                  description="Higher = frames must be more similar to be skipped"
                >
                  <Slider
                    mt={6}
                    value={dedupThreshold}
                    onChange={setDedupThreshold}
                    min={0.8}
                    max={1}
                    step={0.01}
                    disabled={videoBusy}
                    label={(v) => v.toFixed(2)}
                  />
                </Input.Wrapper>
              )}
            </Stack>
          </Collapse>
        </Stack>

        {videoBusy && (
          <Text size="xs" c="dimmed">
            Extraction in progress — see the progress panel (bottom-right). You can leave this page.
          </Text>
        )}

        <Button onClick={start} disabled={!file || videoBusy}>
          Start extraction
        </Button>
      </Stack>
    </Card>
  )
}

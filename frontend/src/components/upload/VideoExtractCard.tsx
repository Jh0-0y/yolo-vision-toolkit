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

const VIDEO_MIME = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm']

/** 추출 파라미터 — 어디로 보낼지는 이 카드가 모른다. */
export interface VideoExtractParams {
  target_fps: number
  max_frames: number
  start_sec: number
  end_sec: number | null
  dedup: boolean
  dedup_threshold: number
}

interface Props {
  /** 서버가 영상 워커를 하나만 돌리므로 한 번에 하나다 — 누가 도는지는 호출자가 안다. */
  busy: boolean
  onStart: (file: File, params: VideoExtractParams) => void
  /** 프레임이 어디로 들어가는지 한 줄 설명 (프로젝트 vs 데이터셋). */
  hint?: string
}

/** 영상에서 프레임을 뽑는 폼. **잡을 직접 던지지 않는다** — 파라미터만 모아 넘긴다.
 *  그래서 학습실 업로드와 데이터셋 가져오기가 같은 폼을 쓴다. */
export default function VideoExtractCard({ busy, onStart, hint }: Props) {
  const videoBusy = busy
  const [file, setFile] = useState<File | null>(null)
  const [targetFps, setTargetFps] = useState<number>(2)
  const [maxFrames, setMaxFrames] = useState<number>(2000)
  const [startSec, setStartSec] = useState<number>(0)
  const [endSec, setEndSec] = useState<number | ''>('')
  const [dedup, setDedup] = useState(true)
  const [dedupThreshold, setDedupThreshold] = useState(0.92)
  const [advanced, setAdvanced] = useState(false)

  const start = () => {
    if (!file) return
    onStart(file, {
      target_fps: targetFps,
      max_frames: maxFrames,
      start_sec: startSec,
      end_sec: endSec === '' ? null : Number(endSec),
      dedup,
      dedup_threshold: dedupThreshold,
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
            <Text fw={600}>Video</Text>
            <Text size="xs" c="dimmed">
              {hint ?? 'Extract frames from a video'}
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

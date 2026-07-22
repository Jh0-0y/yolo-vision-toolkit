import { useState } from 'react'
import {
  Button,
  FileButton,
  Group,
  Modal,
  NumberInput,
  Progress,
  Stack,
  Switch,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useQueryClient } from '@tanstack/react-query'
import {
  subscribeVideoEvents,
  uploadVideo,
  type VideoProgressEvent,
} from '../../api/client'

interface Props {
  projectId: string
  opened: boolean
  onClose: () => void
}

export default function VideoUploadModal({ projectId, opened, onClose }: Props) {
  const queryClient = useQueryClient()
  const [file, setFile] = useState<File | null>(null)
  const [targetFps, setTargetFps] = useState<number>(2)
  const [maxFrames, setMaxFrames] = useState<number>(2000)
  const [dedup, setDedup] = useState(true)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<VideoProgressEvent | null>(null)

  const reset = () => {
    setFile(null)
    setProgress(null)
    setBusy(false)
  }

  const close = () => {
    if (busy) return
    reset()
    onClose()
  }

  const start = async () => {
    if (!file) return
    setBusy(true)
    setProgress(null)
    try {
      const { video_id } = await uploadVideo(projectId, file, {
        target_fps: targetFps,
        max_frames: maxFrames,
        start_sec: 0,
        end_sec: null,
        dedup,
        dedup_threshold: 0.92,
      })
      subscribeVideoEvents(projectId, video_id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          notifications.show({
            message: `프레임 ${ev.saved}장 추출됨${ev.skipped_dup ? `, 중복 ${ev.skipped_dup}장 제외` : ''}`,
            color: 'green',
          })
          queryClient.invalidateQueries({ queryKey: ['images', projectId] })
          queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
          setBusy(false)
        } else if (ev.phase === 'error' || ev.phase === 'cancelled') {
          notifications.show({ message: ev.msg || '추출이 중단되었습니다', color: 'red' })
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
    <Modal opened={opened} onClose={close} title="동영상에서 프레임 추출" centered>
      <Stack>
        <Group>
          <FileButton onChange={setFile} accept="video/*">
            {(props) => (
              <Button {...props} variant="light" disabled={busy}>
                동영상 선택
              </Button>
            )}
          </FileButton>
          <Text size="sm" c={file ? undefined : 'dimmed'} truncate style={{ flex: 1 }}>
            {file ? file.name : 'mp4 · mov · avi · mkv · webm'}
          </Text>
        </Group>

        <Group grow>
          <NumberInput
            label="초당 추출 (fps)"
            description="원본 fps와 무관"
            value={targetFps}
            onChange={(v) => setTargetFps(Number(v) || 1)}
            min={0.1}
            step={0.5}
            disabled={busy}
          />
          <NumberInput
            label="최대 프레임"
            value={maxFrames}
            onChange={(v) => setMaxFrames(Number(v) || 1)}
            min={1}
            disabled={busy}
          />
        </Group>

        <Switch
          label="유사 프레임 제거 (dedup)"
          description="인접한 거의 동일한 프레임을 건너뜁니다"
          checked={dedup}
          onChange={(e) => setDedup(e.currentTarget.checked)}
          disabled={busy}
        />

        {progress && (
          <Stack gap={4}>
            <Progress value={pct} animated={busy} />
            <Text size="xs" c="dimmed">
              {progress.phase === 'done'
                ? `완료 — ${progress.saved}장 추출`
                : progress.phase === 'start'
                  ? `분석 중… (원본 ${progress.src_fps}fps, 총 ${progress.total_frames}프레임)`
                  : `추출 중… ${progress.saved ?? 0}장 (스캔 ${progress.scanned ?? 0}/${progress.total_frames ?? '?'})`}
            </Text>
          </Stack>
        )}

        <Group justify="flex-end">
          <Button variant="default" onClick={close} disabled={busy}>
            닫기
          </Button>
          <Button onClick={start} disabled={!file || busy} loading={busy}>
            추출 시작
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}

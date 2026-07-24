import { useEffect, useRef, useState } from 'react'
import { Alert, Anchor, Badge, Button, Card, Group, Progress, Stack, Text } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconMovie, IconX } from '@tabler/icons-react'
import {
  annotateResultUrl,
  startAnnotate,
  subscribeAnnotateEvents,
  type AnnotateProgress,
} from '../../api/client'
import type { TestConfig } from './TestControls'

const VIDEO_MIME = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo']

export default function AnnotateMode({ projectId, cfg }: { projectId: string; cfg: TestConfig }) {
  const [fileName, setFileName] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<AnnotateProgress | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const unsub = useRef<(() => void) | null>(null)

  useEffect(() => () => unsub.current?.(), [])

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file) return
    if (!cfg.selected.length) {
      setError('모델을 하나 이상 선택하세요')
      return
    }
    unsub.current?.()
    setFileName(file.name)
    setError(null)
    setDone(false)
    setJobId(null)
    setProgress({ phase: 'start' })
    setRunning(true)
    try {
      const { job_id } = await startAnnotate({
        file,
        modelIds: cfg.selected,
        projectId,
        params: { conf: cfg.conf, iou_wbf: cfg.iou, imgsz: Number(cfg.imgsz), device: cfg.device === 'auto' ? null : cfg.device },
      })
      setJobId(job_id)
      unsub.current = subscribeAnnotateEvents(job_id, (ev) => {
        setProgress(ev)
        if (ev.phase === 'done') {
          setDone(true)
          setRunning(false)
        } else if (ev.phase === 'error') {
          setError(ev.msg || '주석 처리 실패')
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
  }

  const pct =
    progress?.total && progress.done != null ? Math.round((progress.done / progress.total) * 100) : 0
  const phaseLabel: Record<string, string> = {
    start: '준비 중…',
    annotate: `추론·박스 그리는 중… ${progress?.done ?? 0}/${progress?.total ?? '?'}`,
    encoding: '영상 인코딩 중…',
    done: '완료',
  }

  return (
    <Card withBorder radius="md" padding="md">
      <Stack gap="md">
        {error && (
          <Alert color="red" icon={<IconAlertTriangle size={18} />} onClose={() => setError(null)} withCloseButton>
            {error}
          </Alert>
        )}

        {!fileName ? (
          <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false}>
            <Stack align="center" gap="xs" py="xl">
              <Dropzone.Idle>
                <IconMovie size={40} stroke={1.2} />
              </Dropzone.Idle>
              <Dropzone.Reject>
                <IconX size={40} />
              </Dropzone.Reject>
              <Text size="sm">동영상(mp4 등)을 드래그하거나 클릭해서 업로드</Text>
              <Text size="xs" c="dimmed">모델이 전체 영상에 박스를 그려 넣은 결과 영상을 만들어 드립니다.</Text>
            </Stack>
          </Dropzone>
        ) : (
          <Stack gap="sm">
            <Group justify="space-between">
              <Text size="sm" truncate="end" maw={360}>{fileName}</Text>
              <Button size="xs" variant="subtle" onClick={reset}>다른 동영상</Button>
            </Group>

            {running && (
              <Stack gap={4}>
                <Group justify="space-between">
                  <Text size="sm">{phaseLabel[progress?.phase ?? 'start'] ?? '처리 중…'}</Text>
                  {progress?.phase === 'annotate' && <Badge variant="light">{pct}%</Badge>}
                </Group>
                <Progress value={progress?.phase === 'encoding' ? 100 : pct} animated />
              </Stack>
            )}

            {done && jobId && (
              <Stack gap="xs">
                {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                <video src={annotateResultUrl(jobId)} controls style={{ width: '100%', borderRadius: 8 }} />
                <Group>
                  <Anchor href={annotateResultUrl(jobId)} download={`annotated_${fileName}`} size="sm">
                    결과 영상 다운로드
                  </Anchor>
                  <Text c="dimmed" size="xs">서버에는 임시 저장되며 1시간 후 자동 삭제됩니다.</Text>
                </Group>
              </Stack>
            )}
          </Stack>
        )}
      </Stack>
    </Card>
  )
}

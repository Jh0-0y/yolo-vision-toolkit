import { useEffect, useMemo, useState } from 'react'
import { Badge, Button, Card, Group, Loader, Slider, Stack, Text } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { notifications } from '@mantine/notifications'
import { IconMovie, IconX } from '@tabler/icons-react'
import {
  predictVideoFrame,
  uploadTestVideo,
  videoFrameUrl,
  type PredictResponse,
  type VideoUpload,
} from '../../api/client'
import type { TestConfig } from './TestControls'
import BoxOverlay from './BoxOverlay'

const VIDEO_MIME = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo']

export default function VideoMode({ projectId, cfg }: { projectId: string; cfg: TestConfig }) {
  const [video, setVideo] = useState<VideoUpload | null>(null)
  const [idx, setIdx] = useState(0)
  const [frameRes, setFrameRes] = useState<{ idx: number; res: PredictResponse } | null>(null)
  const [uploading, setUploading] = useState(false)
  const [predicting, setPredicting] = useState(false)

  async function onDrop(files: File[]) {
    const file = files[0]
    if (!file) return
    setUploading(true)
    setVideo(null)
    setFrameRes(null)
    setIdx(0)
    try {
      const v = await uploadTestVideo(file)
      setVideo(v)
    } catch (e) {
      notifications.show({ color: 'red', message: `업로드/추출 실패: ${(e as Error).message}` })
    } finally {
      setUploading(false)
    }
  }

  // debounced per-frame inference (conf is a client filter → excluded from deps)
  useEffect(() => {
    if (!video || !cfg.selected.length) return
    const t = setTimeout(async () => {
      setPredicting(true)
      try {
        const res = await predictVideoFrame({
          videoId: video.video_id,
          idx,
          modelIds: cfg.selected,
          projectId,
          params: { conf: cfg.conf, iou_wbf: cfg.iou, imgsz: Number(cfg.imgsz), device: cfg.device === 'auto' ? null : cfg.device },
        })
        setFrameRes({ idx, res })
      } catch {
        /* transient scrub errors are ignored */
      } finally {
        setPredicting(false)
      }
    }, 250)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video, idx, cfg.selected, cfg.iou, cfg.imgsz, cfg.device, projectId])

  const boxes = useMemo(() => {
    if (!frameRes || frameRes.idx !== idx) return []
    return frameRes.res.boxes.filter((b) => b.score >= cfg.conf)
  }, [frameRes, idx, cfg.conf])

  if (!video) {
    return (
      <Card withBorder radius="md" padding="md">
        <Dropzone onDrop={onDrop} accept={VIDEO_MIME} multiple={false} loading={uploading}>
          <Stack align="center" gap="xs" py="xl">
            <Dropzone.Idle>
              <IconMovie size={40} stroke={1.2} />
            </Dropzone.Idle>
            <Dropzone.Reject>
              <IconX size={40} />
            </Dropzone.Reject>
            <Text size="sm">동영상(mp4 등)을 드래그하거나 클릭해서 업로드</Text>
            <Text size="xs" c="dimmed">업로드 후 프레임을 추출해 슬라이더로 스크럽하며 추론합니다.</Text>
          </Stack>
        </Dropzone>
      </Card>
    )
  }

  return (
    <Card withBorder radius="md" padding="md">
      <Stack gap="sm">
        <Group justify="space-between">
          <Group gap="xs">
            <Badge variant="light">프레임 {idx + 1} / {video.frame_count}</Badge>
            {predicting && <Loader size="xs" />}
            {frameRes?.idx === idx && (
              <Badge variant="light" color="teal">
                {boxes.length} boxes · {frameRes.res.device}
              </Badge>
            )}
          </Group>
          <Button size="xs" variant="subtle" onClick={() => { setVideo(null); setFrameRes(null) }}>
            다른 동영상
          </Button>
        </Group>

        <BoxOverlay src={videoFrameUrl(video.video_id, idx)} boxes={boxes} showLabels={cfg.showLabels} />

        <Slider
          min={0}
          max={Math.max(0, video.frame_count - 1)}
          step={1}
          value={idx}
          onChange={setIdx}
          label={(v) => `프레임 ${v + 1}`}
        />
        <Text c="dimmed" size="xs">
          슬라이더를 움직이면 해당 프레임을 온디맨드로 추론합니다. Confidence는 즉시 필터링됩니다.
        </Text>
      </Stack>
    </Card>
  )
}

// 크롭 런처 — 옛 Crop JSON · Draw · Video 세 탭을 **하나로 합친 자리**다.
//
// 여기서 정하는 것이 곧 런의 설정 스냅샷이다. 시작하면 `crop_id` 를 받아 상세로
// 넘어가고, 진행률은 거기서 본다(학습이 TrainPage → TrainRunDetailPage 로 가는 것과
// 같다). 이 화면은 인페이지 진행바를 두지 않는다.
//
// 기본값은 **가장 최근 런**에서 출발한다. 따로 저장해 두는 프리셋은 없다 — 마지막에
// 쓴 설정이 곧 다음 런의 출발점이라, 연구가 이어지는 대로 자연히 따라온다.
//
// 영상 아카이브도 **여기서 관리한다** — 따로 탭을 두지 않고 고르는 자리에서 올리고
// 지운다(학습실 TrainPage 의 데이터셋과 같은 형태).
import { useEffect, useState } from 'react'
import {
  ActionIcon,
  Alert,
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
  TextInput,
  Title,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconPlayerPlay, IconTrash, IconVideo } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  DEFAULT_TOGGLES,
  createLabCropRun,
  deleteLabVideo,
  getResources,
  listAllModels,
  listLabCropRuns,
  listLabVideos,
  uploadLabVideo,
  type CropDetector,
  type CropToggles,
  type TrackcropOverrides,
} from '../api/client'
import DetectionSettings, {
  newEntry,
  type DetectorEntry,
} from '../components/lab/DetectionSettings'
import TuningPanel from '../components/lab/TuningPanel'

// 출력 세팅 — 그리기 도구 개별 토글. 하나도 안 켜면 가로 영상은 원본 그대로다.
const TOGGLE_ROWS: { key: keyof CropToggles; label: string; hint: string }[] = [
  { key: 'obj_boxes', label: 'Detection boxes', hint: 'Detected ball and player boxes' },
  { key: 'player_trails', label: 'Player trails', hint: 'Each player’s path over the last 2s' },
  { key: 'ball_trail', label: 'Ball trail', hint: 'The chosen ball’s path over the last 2s' },
  { key: 'target_highlight', label: 'Target highlight', hint: 'The ball and holder the crop picked' },
  { key: 'crop_box', label: 'Crop box', hint: 'Where the cut lands' },
  { key: 'dead_zone', label: 'Dead zone', hint: 'The band where the window stays put' },
  { key: 'center_line', label: 'Center line', hint: 'Aim line — color shows the target type' },
]

function formatBytes(n: number): string {
  if (!n) return '–'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000)
  const m = Math.floor(total / 60)
  const s = String(total % 60).padStart(2, '0')
  return `${m}:${s}`
}

export default function LabCropPage() {
  const navigate = useNavigate()

  const videos = useQuery({
    queryKey: ['lab-videos'],
    queryFn: () => listLabVideos(),
  })
  // 연구실은 프로젝트에 속하지 않는다 — 전체 모델 풀에서 고른다
  const models = useQuery({ queryKey: ['models', 'all'], queryFn: listAllModels })
  const resources = useQuery({
    queryKey: ['resources'],
    queryFn: getResources,
    refetchInterval: 5000,
  })
  // 이 화면의 출발값 — 목록은 최신순이라 첫 항목이 마지막 런이다
  const runs = useQuery({ queryKey: ['lab-crops'], queryFn: listLabCropRuns })

  const [name, setName] = useState('')
  const [videoId, setVideoId] = useState<string | null>(null)
  const [entries, setEntries] = useState<DetectorEntry[]>([newEntry()])
  const [overrides, setOverrides] = useState<TrackcropOverrides>({})
  const [cropW, setCropW] = useState(608)
  const [cropH, setCropH] = useState(1080)
  const [toggles, setToggles] = useState<CropToggles>(DEFAULT_TOGGLES)
  const [seeded, setSeeded] = useState(false) // 출발값을 한 번만 싣는다
  const [percent, setPercent] = useState<number | null>(null)

  // 마지막 런을 이어받는다 — 크기 · 노브 · 모델 · 출력 옵션. 런이 없으면 기본값 그대로.
  // 지워진 모델은 거른다(그 모델로는 시작이 안 된다).
  useEffect(() => {
    if (seeded || !runs.data || !models.data) return
    setSeeded(true)
    const last = runs.data[0]
    if (!last) return
    setCropW(last.crop_w)
    setCropH(last.crop_h)
    setOverrides(last.overrides)
    setToggles({ ...DEFAULT_TOGGLES, ...last.toggles })
    const known = new Set(models.data.map((m) => m.id))
    const kept = last.models.filter((m) => known.has(m.model_id))
    if (kept.length) {
      setEntries(
        kept.map((m) => ({
          ...newEntry(),
          modelId: m.model_id,
          mode: m.mode === 'tiled' ? 'tiled' : 'full',
          conf: m.conf ?? '',
        })),
      )
    }
  }, [runs.data, models.data, seeded])

  const qc = useQueryClient()
  const refreshVideos = () => {
    qc.invalidateQueries({ queryKey: ['lab-videos'] })
    qc.invalidateQueries({ queryKey: ['lab'] })
  }

  // 원본은 기가 단위라 진행률이 필요하다 — 올라오면 바로 그 영상을 고른 상태로 둔다
  const upload = useMutation({
    mutationFn: (file: File) =>
      uploadLabVideo(file, {
        onProgress: setPercent,
        onUploaded: () => setPercent(100), // 업로드는 끝났고 서버가 규격을 읽는 중
      }),
    onSuccess: (v) => {
      notifications.show({ message: `Uploaded: ${v.name}`, color: 'green' })
      setVideoId(v.id)
      refreshVideos()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
    onSettled: () => setPercent(null),
  })

  const removeVideo = useMutation({
    mutationFn: (id: string) => deleteLabVideo(id),
    onSuccess: () => {
      notifications.show({ message: 'Video deleted', color: 'green' })
      setVideoId(null)
      refreshVideos()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const start = useMutation({
    mutationFn: () => {
      const detectors: CropDetector[] = entries
        .filter((e) => e.modelId)
        .map((e) => ({
          model_id: e.modelId as string,
          mode: e.mode,
          conf: e.conf === '' ? null : e.conf,
          imgsz: e.imgsz,
          tile_size: e.tileSize,
          stride: e.stride,
          merge_iou: e.mergeIou,
        }))
      return createLabCropRun({
        name: name.trim(),
        source_video_id: videoId as string,
        detectors,
        overrides,
        crop_w: cropW,
        crop_h: cropH,
        toggles,
      })
    },
    // 시작하면 곧장 상세로 — 진행률은 거기가 보여준다
    onSuccess: (run) => navigate(`/lab/crops/${run.id}`),
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  const videoOptions = (videos.data ?? []).map((v) => ({
    value: v.id,
    label: `${v.name}${v.width ? ` · ${v.width}×${v.height}` : ''}${
      v.duration_ms ? ` · ${formatDuration(v.duration_ms)}` : ''
    }`,
  }))
  const picked = videos.data?.find((v) => v.id === videoId)
  // 소스보다 큰 창은 clamp 된다 — 시작 전에 실제로 나올 크기를 보여 준다
  const applied = picked?.width
    ? {
        w: Math.max(2, Math.min(picked.width, cropW)) & ~1,
        h: Math.max(2, Math.min(picked.height, cropH)) & ~1,
      }
    : null
  const clamped = applied && (applied.w !== cropW || applied.h !== cropH)

  const hasModel = entries.some((e) => e.modelId)
  const overlay = TOGGLE_ROWS.some((r) => toggles[r.key])
  const res = resources.data

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={3}>Crop</Title>
          <Text c="dimmed" size="sm">
            Pick a video, set the models and knobs, and start. Every run produces crop.json, a wide
            video and the vertical cut.
          </Text>
        </div>
        {res && (
          <Badge variant="light" color={res.training_active ? 'orange' : 'gray'}>
            {res.device_label}
          </Badge>
        )}
      </Group>

      {res?.warning && (
        <Alert color="orange" icon={<IconAlertTriangle size={18} />} title="Resource notice">
          {res.warning}
        </Alert>
      )}

      <Card withBorder radius="md" padding="lg">
        <Stack gap="md">
          <TextInput
            label="Test name"
            placeholder={picked?.name ?? 'Named after the video if left empty'}
            value={name}
            onChange={(e) => setName(e.currentTarget.value)}
          />
          <div>
            <Group gap="xs" align="flex-end">
              <Select
                style={{ flex: 1 }}
                label="Video"
                placeholder={
                  videoOptions.length ? 'Pick a video' : 'Drop a video below to get started'
                }
                data={videoOptions}
                value={videoId}
                onChange={setVideoId}
                disabled={!videoOptions.length}
                allowDeselect={false}
              />
              {picked && (
                <ActionIcon
                  color="red"
                  variant="light"
                  size="lg"
                  title="Delete this video from the archive"
                  loading={removeVideo.isPending}
                  onClick={() => {
                    // 런은 자기 사본을 갖는다 — 원본을 지워도 지난 런은 온전하다
                    if (
                      window.confirm(
                        `Delete video "${picked.name}"? Existing crop runs keep their own copies.`,
                      )
                    )
                      removeVideo.mutate(picked.id)
                  }}
                >
                  <IconTrash size={18} />
                </ActionIcon>
              )}
            </Group>
            {picked && (
              <Text size="xs" c="dimmed" mt={4}>
                {formatBytes(picked.size_bytes)}
                {picked.fps ? ` · ${picked.fps.toFixed(1)} fps` : ''}
                {picked.run_count ? ` · used by ${picked.run_count} run(s)` : ' · not used yet'}
              </Text>
            )}
          </div>

          {percent !== null ? (
            <div>
              <Text size="sm" mb={6}>
                {percent < 100 ? `Uploading… ${percent}%` : 'Reading video info…'}
              </Text>
              <Progress value={percent} animated={percent >= 100} />
            </div>
          ) : (
            <Dropzone
              onDrop={(files) => files[0] && upload.mutate(files[0])}
              accept={['video/mp4', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo']}
              multiple={false}
              loading={upload.isPending}
              radius="md"
              p="sm"
            >
              <Group gap="xs" justify="center" style={{ pointerEvents: 'none' }} py={4}>
                <IconVideo size={20} stroke={1.5} />
                <Text size="sm">Add a video — drop one here or click to pick a file</Text>
              </Group>
            </Dropzone>
          )}
        </Stack>
      </Card>

      <Card withBorder radius="md" padding="lg">
        <Stack gap="sm">
          <DetectionSettings
            models={models.data ?? []}
            entries={entries}
            onEntries={setEntries}
            // 샘플링은 노브와 같은 값이다 — 두 곳에 두면 서로 어긋난다
            sampling={overrides.sampling_interval_ms ?? ''}
            onSampling={(v) =>
              setOverrides((o) => {
                const next = { ...o }
                if (v === '') delete next.sampling_interval_ms
                else next.sampling_interval_ms = v
                return next
              })
            }
          />
        </Stack>
      </Card>

      <Card withBorder radius="md" padding="lg">
        <Stack gap="md">
          <div>
            <Text size="sm" fw={600} mb={4}>
              Crop size (px)
            </Text>
            <Group gap="xs" align="center">
              <NumberInput
                aria-label="Crop width"
                value={cropW}
                onChange={(v) => setCropW(Number(v) || 2)}
                min={2}
                step={2}
                w={110}
              />
              <Text c="dimmed">×</Text>
              <NumberInput
                aria-label="Crop height"
                value={cropH}
                onChange={(v) => setCropH(Number(v) || 2)}
                min={2}
                step={2}
                w={110}
              />
              {applied && (
                <Text size="xs" c={clamped ? 'orange' : 'dimmed'}>
                  {clamped
                    ? `→ clamped to ${applied.w}×${applied.h} (source is ${picked?.width}×${picked?.height})`
                    : `from ${picked?.width}×${picked?.height}`}
                </Text>
              )}
            </Group>
          </div>
          <TuningPanel
            value={overrides}
            onChange={setOverrides}
            exclude={['sampling_interval_ms']}
          />
        </Stack>
      </Card>

      <Card withBorder radius="md" padding="lg">
        <Stack gap="xs">
          <Text size="sm" fw={600}>
            Output
          </Text>
          <Text size="xs" c="dimmed">
            {overlay
              ? 'The wide video is rendered with these overlays burned in.'
              : 'Nothing to draw — the wide video is the original, hard-linked (no extra disk).'}
          </Text>
          {TOGGLE_ROWS.map((row) => (
            <Checkbox
              key={row.key}
              size="sm"
              label={
                <Group gap={6}>
                  <Text size="sm">{row.label}</Text>
                  <Text size="xs" c="dimmed">
                    {row.hint}
                  </Text>
                </Group>
              }
              checked={toggles[row.key]}
              // 이벤트 값은 **여기서** 읽는다. 업데이터 함수 안에서 읽으면 React 가
              // 그 함수를 나중에 부르는 사이 currentTarget 이 null 이 되어 앱이 죽는다.
              onChange={(e) => {
                const checked = e.currentTarget.checked
                setToggles((t) => ({ ...t, [row.key]: checked }))
              }}
            />
          ))}
        </Stack>
      </Card>

      <Group justify="flex-end">
        <Button
          size="md"
          leftSection={<IconPlayerPlay size={18} />}
          loading={start.isPending}
          disabled={!videoId || !hasModel}
          onClick={() => start.mutate()}
        >
          Start crop run
        </Button>
      </Group>
    </Stack>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Checkbox,
  Group,
  Progress,
  Stack,
  Text,
  UnstyledButton,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconChevronRight, IconMovie, IconRefresh, IconX } from '@tabler/icons-react'
import {
  getLiveResult,
  livePlan,
  liveVideoUrl,
  type CropPlan,
  type LiveResult,
  type ModelOut,
  type TrackcropOverrides,
} from '../../api/client'
import DetectionSettings from './DetectionSettings'
import TuningPanel from './TuningPanel'
import { DEFAULT_IMGSZ } from './useAnnotateJob'
import { LIVE_PHASE_LABEL, useLiveJob } from './useLiveJob'
import { drawOverlay } from './liveOverlay'

const VIDEO_MIME = [
  'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo',
]

interface Props {
  projectId: string
  models: ModelOut[]
}

/** Live Crop Preview — detect once, then play the video with the crop overlay drawn on a
 *  canvas. Tuning knobs recompute the crop trajectory instantly (no re-inference). */
export default function CropDrawMode({ projectId, models }: Props) {
  // detection params (a change needs a re-detect)
  const [modelId, setModelId] = useState<string | null>(null)
  const [conf, setConf] = useState<number | ''>('')
  const [imgsz, setImgsz] = useState(DEFAULT_IMGSZ)
  const [sampling, setSampling] = useState<number | ''>('')
  // tuning overrides (recomputed on the fly)
  const [overrides, setOverrides] = useState<TrackcropOverrides>({})
  // overlay toggles (pure draw switches, except highlight which needs the debug pass)
  const [objInference, setObjInference] = useState(true)
  const [drawCropBox, setDrawCropBox] = useState(true)
  const [showDeadZone, setShowDeadZone] = useState(true)
  const [showCenterLine, setShowCenterLine] = useState(true)
  const [showHighlight, setShowHighlight] = useState(true)
  const [showOverlays, setShowOverlays] = useState(false)

  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<LiveResult | null>(null)
  const [plan, setPlan] = useState<CropPlan | null>(null)
  const [planLoading, setPlanLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [videoFailed, setVideoFailed] = useState(false)
  const [analyzedKey, setAnalyzedKey] = useState<string | null>(null)

  const job = useLiveJob((detectId) => {
    getLiveResult(detectId).then(setResult).catch((e) => setLoadError((e as Error).message))
  })

  useEffect(() => {
    if (models.length && !modelId) setModelId(models[0].id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

  const detectionKey = JSON.stringify({ modelId, conf, imgsz, sampling })
  const detectionDirty = analyzedKey != null && analyzedKey !== detectionKey
  const deadZoneWidth = overrides.dead_zone_width ?? 208

  async function analyze(f: File) {
    if (!modelId) return
    setLoadError(null)
    setVideoFailed(false)
    setResult(null)
    setPlan(null)
    setFile(f)
    setAnalyzedKey(detectionKey)
    await job.run({
      file: f,
      modelIds: [modelId],
      projectId,
      conf: conf === '' ? undefined : conf,
      imgsz,
      device: null,
      samplingIntervalMs: sampling === '' ? undefined : sampling,
    })
  }

  function reset() {
    job.reset()
    setFile(null)
    setResult(null)
    setPlan(null)
    setAnalyzedKey(null)
    setLoadError(null)
    setVideoFailed(false)
  }

  /** 현재 튜닝이 반영된 plan(CropResult)을 crop.json 파일로 저장한다. */
  function downloadPlan() {
    if (!plan || !file) return
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(plan, null, 2)], { type: 'application/json' }),
    )
    const a = document.createElement('a')
    a.href = url
    a.download = `crop_${file.name.replace(/\.[^.]+$/, '')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // recompute the crop trajectory whenever tuning / highlight / the detection cache changes.
  // debounced + stale-drop so dragging a knob only applies the latest request.
  const planReq = useRef(0)
  const overridesKey = JSON.stringify(overrides)
  useEffect(() => {
    if (!result) return
    const id = ++planReq.current
    setPlanLoading(true)
    const timer = window.setTimeout(async () => {
      try {
        const p = await livePlan(result.detect_id, overrides, showHighlight)
        if (planReq.current === id) setPlan(p)
      } catch (e) {
        if (planReq.current === id) setLoadError((e as Error).message)
      } finally {
        if (planReq.current === id) setPlanLoading(false)
      }
    }, 180)
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overridesKey, showHighlight, result?.detect_id])

  // ---- canvas overlay render loop (reads latest state from a ref, no re-subscribe) ----
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detOffsets = useMemo(
    () => (result ? result.detections.map((s) => s.video_offset_ms) : []),
    [result],
  )
  const drawRef = useRef<{
    result: LiveResult | null
    plan: CropPlan | null
    detOffsets: number[]
    toggles: {
      objInference: boolean
      drawCropBox: boolean
      showDeadZone: boolean
      showCenterLine: boolean
      showHighlight: boolean
    }
    deadZoneWidth: number
  }>({ result: null, plan: null, detOffsets: [], toggles: {
    objInference, drawCropBox, showDeadZone, showCenterLine, showHighlight,
  }, deadZoneWidth })
  drawRef.current = {
    result, plan, detOffsets,
    toggles: { objInference, drawCropBox, showDeadZone, showCenterLine, showHighlight },
    deadZoneWidth,
  }

  useEffect(() => {
    if (!result) return
    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas) return
      const cw = canvas.clientWidth
      const ch = canvas.clientHeight
      if (!cw || !ch) return
      // back the canvas at device-pixel resolution so lines stay crisp on retina
      const dpr = window.devicePixelRatio || 1
      const bw = Math.round(cw * dpr)
      const bh = Math.round(ch * dpr)
      if (canvas.width !== bw) canvas.width = bw
      if (canvas.height !== bh) canvas.height = bh
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0) // draw in CSS pixels
      ctx.clearRect(0, 0, cw, ch)
      const st = drawRef.current
      if (!st.result) return
      drawOverlay({
        ctx,
        canvasW: cw,
        canvasH: ch,
        sourceW: st.result.source_width,
        sourceH: st.result.source_height,
        ms: video.currentTime * 1000,
        detections: st.result.detections,
        detOffsets: st.detOffsets,
        plan: st.plan,
        toggles: st.toggles,
        deadZoneWidth: st.deadZoneWidth,
      })
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [result])

  const error = job.error || loadError

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <DetectionSettings
            models={models}
            modelId={modelId}
            onModelId={setModelId}
            imgsz={imgsz}
            onImgsz={setImgsz}
            conf={conf}
            onConf={setConf}
            sampling={sampling}
            onSampling={setSampling}
            disabled={job.running}
            description="Runs once on upload. Changing these re-analyzes; tuning below applies instantly."
          />

          {detectionDirty && file && !job.running && (
            <Button
              variant="light"
              leftSection={<IconRefresh size={16} />}
              onClick={() => analyze(file)}
            >
              Re-analyze with new detection settings
            </Button>
          )}

          <TuningPanel
            value={overrides}
            onChange={setOverrides}
            disabled={job.running}
            exclude={['sampling_interval_ms']}
          />

          <Stack gap={4}>
            <UnstyledButton onClick={() => setShowOverlays((v) => !v)}>
              <Group gap={4}>
                <IconChevronRight
                  size={14}
                  style={{ transform: showOverlays ? 'rotate(90deg)' : 'none', transition: 'transform 150ms' }}
                />
                <Text size="sm" fw={600}>Overlays</Text>
              </Group>
            </UnstyledButton>
            {showOverlays && (
              <Stack gap={4} pl="lg">
                <Checkbox
                  label="Object inference — ByteTrack boxes, IDs & conf"
                  checked={objInference}
                  onChange={(e) => setObjInference(e.currentTarget.checked)}
                  size="xs"
                />
                <Checkbox
                  label="Crop tracking box — 9:16 rectangle"
                  checked={drawCropBox}
                  onChange={(e) => setDrawCropBox(e.currentTarget.checked)}
                  size="xs"
                />
                {drawCropBox && (
                  <Stack gap={4} pl="lg">
                    <Checkbox
                      label="Dead zone band"
                      checked={showDeadZone}
                      onChange={(e) => setShowDeadZone(e.currentTarget.checked)}
                      size="xs"
                    />
                    <Checkbox
                      label="Center line & type label"
                      checked={showCenterLine}
                      onChange={(e) => setShowCenterLine(e.currentTarget.checked)}
                      size="xs"
                    />
                  </Stack>
                )}
                <Checkbox
                  label="Target highlight — selected ball (▼) & carrier markers"
                  checked={showHighlight}
                  onChange={(e) => setShowHighlight(e.currentTarget.checked)}
                  size="xs"
                />
              </Stack>
            )}
          </Stack>
        </Stack>
      </Card>

      <Card withBorder radius="md" padding="md">
        <Stack gap="md">
          {error && (
            <Alert color="red" icon={<IconAlertTriangle size={18} />} withCloseButton onClose={() => { job.setError(null); setLoadError(null) }}>
              {error}
            </Alert>
          )}

          {!file ? (
            <Dropzone onDrop={(files) => files[0] && analyze(files[0])} accept={VIDEO_MIME} multiple={false} disabled={!modelId}>
              <Stack align="center" gap="xs" py="xl">
                <Dropzone.Idle><IconMovie size={40} stroke={1.2} /></Dropzone.Idle>
                <Dropzone.Reject><IconX size={40} /></Dropzone.Reject>
                <Text size="sm">Drop a video (mp4, mov, …) or click to upload</Text>
                <Text size="xs" c="dimmed">Detects once, then plays back with a live crop overlay.</Text>
              </Stack>
            </Dropzone>
          ) : (
            <Stack gap="sm">
              <Group justify="space-between">
                <Text size="sm" truncate="end" maw={360}>{file.name}</Text>
                <Group gap="xs">
                  {planLoading && result && <Badge variant="light" color="blue">updating…</Badge>}
                  <Button size="xs" variant="subtle" onClick={reset} disabled={job.running}>New video</Button>
                </Group>
              </Group>

              {job.running && (
                <Stack gap={4}>
                  <Group justify="space-between">
                    <Text size="sm">{LIVE_PHASE_LABEL[job.progress?.phase ?? 'start'] ?? 'Working…'}</Text>
                    {job.progress?.phase === 'detect' && <Badge variant="light">{job.pct}%</Badge>}
                  </Group>
                  <Progress value={job.progress?.phase === 'encoding' ? 100 : job.pct} animated />
                </Stack>
              )}

              {result && !job.running && (
                videoFailed ? (
                  <Alert color="orange" icon={<IconAlertTriangle size={18} />}>
                    The preview couldn't play inline (unsupported codec in this browser).
                  </Alert>
                ) : (
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
                      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                      <video
                        ref={videoRef}
                        key={result.detect_id}
                        src={liveVideoUrl(result.detect_id)}
                        controls
                        onError={() => setVideoFailed(true)}
                        style={{ display: 'block', maxWidth: '100%', maxHeight: '70vh', borderRadius: 8 }}
                      />
                      <canvas
                        ref={canvasRef}
                        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                      />
                    </div>
                  </div>
                )
              )}

              {plan && result && !job.running && (
                <Group>
                  <Anchor size="sm" onClick={() => downloadPlan()} style={{ cursor: 'pointer' }}>
                    Download crop coordinates (JSON) — current tuning
                  </Anchor>
                  <Text c="dimmed" size="xs">Reflects the knobs above; re-download after changes.</Text>
                </Group>
              )}
            </Stack>
          )}
        </Stack>
      </Card>
    </Stack>
  )
}

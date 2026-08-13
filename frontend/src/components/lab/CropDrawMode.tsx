import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Checkbox,
  Group,
  Stack,
  Text,
  UnstyledButton,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconAlertTriangle, IconChevronRight, IconMovie, IconRefresh, IconX } from '@tabler/icons-react'
import {
  getLiveResult,
  getLiveStatus,
  livePlan,
  liveRenderVideoUrl,
  liveVideoUrl,
  type CropPlan,
  type DetectorPayload,
  type LiveResult,
  type ModelOut,
  type TrackcropOverrides,
} from '../../api/client'
import DetectionSettings, { type DetectorEntry } from './DetectionSettings'
import TuningPanel from './TuningPanel'
import { clearSession, loadSession, saveSession } from './cropDrawSession'
import { useLiveJob } from './useLiveJob'
import { drawOverlay } from './liveOverlay'

function entryPayload(entries: DetectorEntry[]): DetectorPayload[] {
  return entries.filter((e) => e.modelId).map((e) => ({
    model_id: e.modelId as string,
    mode: e.mode,
    conf: e.conf === '' ? undefined : e.conf,
    imgsz: e.imgsz,
    tile_size: e.tileSize,
    stride: e.stride,
    merge_iou: e.mergeIou,
  }))
}

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
  // 지난 세션 — 탭을 옮기거나 새로고침해도 여기서 되살린다
  const [saved] = useState(() => loadSession(projectId))

  // detection params (a change needs a re-detect)
  const [entries, setEntries] = useState<DetectorEntry[]>(saved.entries)
  const [sampling, setSampling] = useState<number | ''>(saved.sampling)
  const base = entries[0]
  // tuning overrides (recomputed on the fly)
  const [overrides, setOverrides] = useState<TrackcropOverrides>(saved.overrides)
  // overlay toggles (pure draw switches, except highlight which needs the debug pass)
  const [objInference, setObjInference] = useState(saved.toggles.objInference)
  const [showTrails, setShowTrails] = useState(saved.toggles.showTrails)
  const [drawCropBox, setDrawCropBox] = useState(saved.toggles.drawCropBox)
  const [showDeadZone, setShowDeadZone] = useState(saved.toggles.showDeadZone)
  const [showCenterLine, setShowCenterLine] = useState(saved.toggles.showCenterLine)
  const [showHighlight, setShowHighlight] = useState(saved.toggles.showHighlight)
  const [showOverlays, setShowOverlays] = useState(false)

  // 원본 File 은 이 브라우저 세션 안에서만 있다 — 재분석에만 쓴다. 되살린 세션에는
  // 이름만 남아 있고, 영상은 서버 프리뷰(liveVideoUrl)로 다시 재생한다.
  const [file, setFile] = useState<File | null>(null)
  const [fileName, setFileName] = useState<string | null>(saved.fileName)
  const [detectId, setDetectId] = useState<string | null>(saved.detectId)
  const [renderJobId, setRenderJobId] = useState<string | null>(saved.renderJobId)
  const [restoring, setRestoring] = useState(saved.detectId != null)
  const [result, setResult] = useState<LiveResult | null>(null)
  const [plan, setPlan] = useState<CropPlan | null>(null)
  const [planLoading, setPlanLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [videoFailed, setVideoFailed] = useState(false)
  const [analyzedKey, setAnalyzedKey] = useState<string | null>(saved.analyzedKey)
  // 두 서버 잡 모두 전역 잡 카드가 진행률을 보여준다 — 이 화면은 결과만 받는다.
  const detectJob = useLiveJob(projectId) // 검출 패스 (모델·GPU)
  const renderJob = useLiveJob(projectId) // 오버레이 렌더 (추론 없음 — 캐시로 영상 굽기)
  const busy = detectJob.starting || detectJob.status === 'running'
  // 카드는 사용자가 닫을 수 있다 — 끝났다는 사실은 여기 걸어 둔다
  const [detectDone, setDetectDone] = useState(false)
  const [renderReady, setRenderReady] = useState(false)

  useEffect(() => {
    if (detectJob.status === 'done') setDetectDone(true)
  }, [detectJob.status, detectJob.seq])

  useEffect(() => {
    if (renderJob.status === 'done') setRenderReady(true)
  }, [renderJob.status, renderJob.seq])

  // 검출이 끝나면 캐시를 받아 온다
  useEffect(() => {
    if (!detectDone || !detectId || result?.detect_id === detectId) return
    let alive = true
    getLiveResult(detectId)
      .then((r) => alive && setResult(r))
      .catch((e) => alive && setLoadError((e as Error).message))
    return () => {
      alive = false
    }
  }, [detectDone, detectId, result?.detect_id])

  useEffect(() => {
    if (models.length && !entries[0].modelId)
      setEntries((prev) => [{ ...prev[0], modelId: models[0].id }, ...prev.slice(1)])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

  // 화면에 남아 있는 것을 그대로 적어 둔다 — 검출 결과는 담지 않는다(서버가 갖고 있다).
  useEffect(() => {
    saveSession({
      projectId,
      detectId,
      renderJobId,
      fileName,
      analyzedKey,
      entries,
      sampling,
      overrides,
      toggles: { objInference, showTrails, drawCropBox, showDeadZone, showCenterLine, showHighlight },
    })
  }, [
    projectId, detectId, renderJobId, fileName, analyzedKey, entries, sampling, overrides,
    objInference, showTrails, drawCropBox, showDeadZone, showCenterLine, showHighlight,
  ])

  /** 프리뷰만 놓아준다 — 튜닝 노브는 남긴다. 어차피 다시 쓸 값이다. */
  function forgetPreview(why: string | null) {
    setDetectId(null)
    setRenderJobId(null)
    setFileName(null)
    setAnalyzedKey(null)
    setResult(null)
    setPlan(null)
    setNotice(why)
  }

  // 기억해 둔 세션 되살리기. 캐시가 아직 살아 있는지는 서버만 안다 — TTL 로 쓸려 나간다.
  useEffect(() => {
    if (!saved.detectId) return
    const id = saved.detectId
    let alive = true
    getLiveStatus(id)
      .then(async ({ status, msg, has_render }) => {
        if (!alive) return
        if (has_render) setRenderReady(true)
        // 렌더 잡은 SSE 가 처음부터 재생되므로 상태를 따로 묻지 않고 카드에 올리면 된다
        if (saved.renderJobId) renderJob.attach(saved.renderJobId, `Overlay — ${saved.fileName ?? 'video'}`)
        if (status === 'done') {
          const r = await getLiveResult(id)
          if (alive) setResult(r)
        } else if (status === 'running') {
          // 도는 중이면 전역 카드에 올린다 — 진행률은 처음부터 재생된다
          detectJob.attach(id, saved.fileName ?? 'Crop Draw')
        } else if (status === 'expired') {
          forgetPreview('The previous preview expired — its cache is cleared after an hour.')
        } else {
          forgetPreview(msg || 'The previous detection did not finish.')
        }
      })
      .catch(() => alive && forgetPreview('Could not restore the previous preview.'))
      .finally(() => alive && setRestoring(false))
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 프로젝트가 바뀌면 이전 세션은 남의 것이다
  const mountedProject = useRef(projectId)
  useEffect(() => {
    if (mountedProject.current === projectId) return
    mountedProject.current = projectId
    reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  const detectionKey = JSON.stringify({ entries, sampling })
  const detectionDirty = analyzedKey != null && analyzedKey !== detectionKey
  const deadZoneWidth = overrides.dead_zone_width ?? 208

  async function analyze(f: File) {
    if (!base.modelId) return
    setLoadError(null)
    setNotice(null)
    setVideoFailed(false)
    setResult(null)
    setPlan(null)
    setDetectId(null)
    setFile(f)
    setFileName(f.name)
    setAnalyzedKey(detectionKey)
    setDetectDone(false)
    setRenderReady(false)
    setRenderJobId(null)
    renderJob.reset()
    // startLive 가 돌려주는 job_id 가 곧 detect_id 다 — 캐시도 이 이름으로 찾는다
    setDetectId(
      await detectJob.run({
        file: f,
        modelIds: [base.modelId],
        projectId,
        conf: base.conf === '' ? undefined : base.conf,
        imgsz: base.imgsz,
        device: null,
        samplingIntervalMs: sampling === '' ? undefined : sampling,
        detectors: entryPayload(entries),
      }),
    )
  }

  function reset() {
    detectJob.reset()
    renderJob.reset()
    setDetectDone(false)
    setRenderReady(false)
    setRenderJobId(null)
    setFile(null)
    setFileName(null)
    setDetectId(null)
    setResult(null)
    setPlan(null)
    setAnalyzedKey(null)
    setLoadError(null)
    setNotice(null)
    setVideoFailed(false)
    setRestoring(false)
    clearSession()
  }

  /** 오버레이 영상 렌더 시작 — 검출 캐시 + 현재 튜닝/토글로 서버가 굽는다.
   *  진행률은 전역 잡 카드가 보여주므로 여기서는 시작만 한다. */
  async function renderVideo() {
    if (!result) return
    setRenderJobId(
      await renderJob.render(result.detect_id, `Overlay — ${fileName ?? 'video'}`, overrides, {
        obj_boxes: objInference,
        show_trails: showTrails,
        draw_crop_box: drawCropBox,
        show_dead_zone: showDeadZone,
        show_center_line: showCenterLine,
        show_highlight: showHighlight,
      }),
    )
  }

  /** 현재 튜닝이 반영된 plan을 계약 스키마(camelCase) crop.json으로 저장한다.
   *  samples/debug는 오버레이용 내부 확장이라 내보내기에서 제거한다. */
  function downloadPlan() {
    if (!plan || !fileName) return
    const { samples: _samples, debug: _debug, ...spec } = plan
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(spec, null, 2)], { type: 'application/json' }),
    )
    const a = document.createElement('a')
    a.href = url
    a.download = `crop_${fileName.replace(/\.[^.]+$/, '')}.json`
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
      showTrails: boolean
      drawCropBox: boolean
      showDeadZone: boolean
      showCenterLine: boolean
      showHighlight: boolean
    }
    deadZoneWidth: number
  }>({ result: null, plan: null, detOffsets: [], toggles: {
    objInference, showTrails, drawCropBox, showDeadZone, showCenterLine, showHighlight,
  }, deadZoneWidth })
  drawRef.current = {
    result, plan, detOffsets,
    toggles: { objInference, showTrails, drawCropBox, showDeadZone, showCenterLine, showHighlight },
    deadZoneWidth,
  }

  useEffect(() => {
    if (!result) return
    let stopped = false
    let raf = 0
    let vfc = 0
    let synced: HTMLVideoElement | null = null // rVFC를 건 video (지원 시)

    /** 화면에 표시 중인 프레임의 시각(ms) 기준으로 오버레이를 다시 그린다. */
    const paint = (ms: number) => {
      const canvas = canvasRef.current
      if (!canvas) return
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
        ms,
        detections: st.result.detections,
        detOffsets: st.detOffsets,
        plan: st.plan,
        toggles: st.toggles,
        deadZoneWidth: st.deadZoneWidth,
      })
    }

    // 재생 중에는 requestVideoFrameCallback으로 그린다. 콜백이 주는 mediaTime은
    // "지금 화면에 올라간 그 프레임"의 시각이라, currentTime 폴링과 달리 오버레이가
    // 한 프레임 뒤처지지 않는다 (빠른 공 기준 수십 px 오차의 원인).
    const onFrame: VideoFrameRequestCallback = (_now, meta) => {
      if (stopped || !synced) return
      paint(meta.mediaTime * 1000)
      vfc = synced.requestVideoFrameCallback(onFrame)
    }

    // rAF는 (a) video 마운트 대기, (b) rVFC 미지원 폴백, (c) 일시정지·탐색·리사이즈
    // 중 갱신을 맡는다. 재생 중 rVFC가 그리고 있을 때는 중복으로 그리지 않는다.
    const tick = () => {
      if (stopped) return
      raf = requestAnimationFrame(tick)
      const video = videoRef.current
      if (!video) return
      if (!synced && 'requestVideoFrameCallback' in video) {
        synced = video
        vfc = video.requestVideoFrameCallback(onFrame)
      }
      if (!synced || video.paused || video.seeking) paint(video.currentTime * 1000)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      stopped = true
      cancelAnimationFrame(raf)
      if (synced && vfc) synced.cancelVideoFrameCallback(vfc)
    }
  }, [result])

  const error = detectJob.error || renderJob.error || loadError

  return (
    <Stack gap="md">
      <Card withBorder radius="md" padding="md">
        <Stack gap="sm">
          <DetectionSettings
            models={models}
            entries={entries}
            onEntries={setEntries}
            sampling={sampling}
            onSampling={setSampling}
            disabled={busy}
          />

          {detectionDirty && !busy && (
            file ? (
              <Button
                variant="light"
                leftSection={<IconRefresh size={16} />}
                onClick={() => analyze(file)}
              >
                Re-analyze with new detection settings
              </Button>
            ) : (
              // 되살린 세션에는 원본 File 이 없다 — 다시 올려야 재분석할 수 있다
              <Text size="xs" c="dimmed">
                Detection settings changed. Drop the video again to re-analyze.
              </Text>
            )
          )}

          <TuningPanel
            value={overrides}
            onChange={setOverrides}
            disabled={busy}
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
                  label="Motion trails — recent 2s path per player & selected ball"
                  checked={showTrails}
                  onChange={(e) => setShowTrails(e.currentTarget.checked)}
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
            <Alert color="red" icon={<IconAlertTriangle size={18} />} withCloseButton onClose={() => { detectJob.setError(null); renderJob.setError(null); setLoadError(null) }}>
              {error}
            </Alert>
          )}

          {notice && (
            <Alert color="orange" icon={<IconAlertTriangle size={18} />} withCloseButton onClose={() => setNotice(null)}>
              {notice}
            </Alert>
          )}

          {!fileName ? (
            <Dropzone onDrop={(files) => files[0] && analyze(files[0])} accept={VIDEO_MIME} multiple={false} disabled={!base.modelId}>
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
                <Text size="sm" truncate="end" maw={360}>{fileName}</Text>
                <Group gap="xs">
                  {planLoading && result && <Badge variant="light" color="blue">updating…</Badge>}
                  <Button size="xs" variant="subtle" onClick={reset} disabled={busy}>New video</Button>
                </Group>
              </Group>

              {restoring && !busy && (
                <Text size="sm" c="dimmed">Restoring the previous preview…</Text>
              )}

              {busy && (
                <Text size="sm" c="dimmed">
                  Detecting in the background — progress is in the job card, top right.
                </Text>
              )}

              {result && !busy && (
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

              {plan && result && !busy && (
                <Group gap="md">
                  <Anchor size="sm" onClick={() => downloadPlan()} style={{ cursor: 'pointer' }}>
                    Download crop.json — current tuning
                  </Anchor>
                  {renderJob.status === 'running' ? (
                    <Text size="sm" c="dimmed">Rendering overlay video — see the job card</Text>
                  ) : (
                    <Anchor size="sm" onClick={() => void renderVideo()} style={{ cursor: 'pointer' }}>
                      {renderReady ? 'Re-render overlay video' : 'Render overlay video'}
                    </Anchor>
                  )}
                  {renderReady && (
                    <Anchor size="sm" href={liveRenderVideoUrl(result.detect_id)} download="crop_overlay.mp4">
                      Download overlay video
                    </Anchor>
                  )}
                </Group>
              )}
            </Stack>
          )}
        </Stack>
      </Card>
    </Stack>
  )
}

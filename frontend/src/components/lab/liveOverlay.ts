// Canvas overlay drawing for the live crop preview.
//
// Mirrors the backend geometry (app/domain/crop_render.py + adaptive-crop) so the box
// the client draws matches exactly what a batch render would burn in: 9:16 crop
// width from the frame height, linear-interpolated target centre, edge clamping.
// All trajectory coordinates are in SOURCE pixels; we scale to the displayed canvas.
//
// Detection is sampled at ~100ms, so ByteTrack boxes and highlight markers are
// interpolated between samples (matched by track_id) — otherwise they'd visibly
// jump at 10fps under a 30fps video.

import type { CropPlan, LiveDetection, LiveSample } from '../../api/client'

// Track colours — RGB mirror of the worker's BGR palette (annotate_worker._PALETTE).
const PALETTE = [
  '#4dabf7', '#51cf66', '#ff6b6b', '#ffd43b', '#cc5de8',
  '#20c997', '#ff922b', '#f783ac', '#748ffc', '#a9e34b',
]
const TYPE_COLORS: Record<string, string> = {
  ball: '#ff5c5c',
  ball_player: '#3cc83c',
  player_group: '#3cbeff',
  center: '#a0a0a0',
}
const CROP_COLOR = '#ffe600' // yellow — the crop window
const DEADZONE_COLOR = 'rgba(220,220,220,0.8)'
const TARGET_COLOR = '#ffa500' // orange — target centre line
const BALL_HL = '#ff5c5c'
const CARRIER_HL = '#3cdc3c'

const trackColor = (key: number) => PALETTE[((key % PALETTE.length) + PALETTE.length) % PALETTE.length]
const lerp = (a: number, b: number, t: number) => a + (b - a) * t

/** Even 9:16 crop width for a frame, clamped to the frame (mirror crop_width_for). */
export function cropWidthFor(height: number, frameWidth: number): number {
  let w = Math.max(2, Math.round((height * 9) / 16))
  w = Math.min(w, frameWidth)
  if (w % 2) w -= 1
  return Math.max(2, w)
}

/** Left X of a crop_w window centred on cx, clamped inside the frame (mirror _left_edge). */
function leftEdge(cx: number, cropW: number, frameWidth: number): number {
  const left = Math.round(cx - cropW / 2)
  return Math.max(0, Math.min(left, frameWidth - cropW))
}

/** Largest index whose offset <= ms (step lookup), or -1. Offsets are ascending. */
function stepIndexAt(offsets: number[], ms: number): number {
  let lo = 0
  let hi = offsets.length - 1
  let res = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (offsets[mid] <= ms) {
      res = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return res
}

/** Linear-interpolate the target centre X at `ms` (mirror center_at + build_trajectory). */
function interpCenter(samples: CropPlan['samples'], ms: number, sourceWidth: number): number {
  if (!samples.length) return sourceWidth / 2
  const cxAt = (s: CropPlan['samples'][number]) =>
    s.target_type === 'center' ? sourceWidth / 2 : s.target_center_x
  if (ms <= samples[0].video_offset_ms) return cxAt(samples[0])
  const last = samples[samples.length - 1]
  if (ms >= last.video_offset_ms) return cxAt(last)
  let lo = 0
  let hi = samples.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (samples[mid].video_offset_ms <= ms) lo = mid + 1
    else hi = mid
  }
  const s0 = samples[lo - 1]
  const s1 = samples[lo]
  const t0 = s0.video_offset_ms
  const t1 = s1.video_offset_ms
  const x0 = cxAt(s0)
  const x1 = cxAt(s1)
  return t1 === t0 ? x0 : x0 + (x1 - x0) * ((ms - t0) / (t1 - t0))
}

/** Detections at `ms`, interpolated between the two bracketing samples by track_id so
 *  boxes glide instead of stepping at the 100ms sampling rate. */
function interpDetections(detections: LiveSample[], offsets: number[], ms: number): LiveDetection[] {
  if (!detections.length) return []
  const i = stepIndexAt(offsets, ms)
  if (i < 0) return detections[0].detections
  if (i >= detections.length - 1) return detections[detections.length - 1].detections
  const a = detections[i].detections
  const b = detections[i + 1].detections
  const t0 = offsets[i]
  const t1 = offsets[i + 1]
  const t = t1 > t0 ? (ms - t0) / (t1 - t0) : 0

  const bById = new Map<number, LiveDetection>()
  for (const d of b) if (d.track_id != null) bById.set(d.track_id, d)
  const usedBIdx = new Set<number>() // b 배열 인덱스 기준 사용 표시 (id 유무 무관)

  const lerpDet = (d: LiveDetection, e: LiveDetection): LiveDetection => ({
    ...d,
    bbox_x: lerp(d.bbox_x, e.bbox_x, t),
    bbox_y: lerp(d.bbox_y, e.bbox_y, t),
    bbox_width: lerp(d.bbox_width, e.bbox_width, t),
    bbox_height: lerp(d.bbox_height, e.bbox_height, t),
  })

  const out: LiveDetection[] = []
  for (const d of a) {
    // 1순위: track_id 매칭 (선수 — ByteTrack)
    const e = d.track_id != null ? bById.get(d.track_id) : undefined
    if (e) {
      usedBIdx.add(b.indexOf(e))
      out.push(lerpDet(d, e))
      continue
    }
    // 2순위: id 없는 검출(타일/predict 공) — 같은 타입의 미사용 최근접과 매칭.
    // 이게 없으면 공 박스가 보간되지 못해 100ms마다 툭툭 점프한다.
    let bestIdx = -1
    let bestDist = 250 // px (source) — 이 이상 떨어지면 다른 물체로 본다
    const cx = d.bbox_x + d.bbox_width / 2
    const cy = d.bbox_y + d.bbox_height / 2
    for (let j = 0; j < b.length; j++) {
      if (usedBIdx.has(j) || b[j].object_type !== d.object_type) continue
      const ex = b[j].bbox_x + b[j].bbox_width / 2
      const ey = b[j].bbox_y + b[j].bbox_height / 2
      const dist = Math.hypot(ex - cx, ey - cy)
      if (dist < bestDist) {
        bestDist = dist
        bestIdx = j
      }
    }
    if (bestIdx >= 0) {
      usedBIdx.add(bestIdx)
      out.push(lerpDet(d, b[bestIdx]))
    } else if (t < 0.5) {
      out.push(d) // track leaving — hold through the first half of the gap
    }
  }
  // 다음 샘플에서 새로 나타나는 검출 — 후반부에 등장
  for (let j = 0; j < b.length; j++) {
    if (!usedBIdx.has(j) && t >= 0.5) out.push(b[j])
  }
  return out
}

/** Interpolate a single bbox between two samples ([x,y,w,h]); falls back to `a`. */
function lerpBbox(
  a: [number, number, number, number] | null,
  b: [number, number, number, number] | null,
  t: number,
): [number, number, number, number] | null {
  if (!a) return b && t >= 0.5 ? b : null
  if (!b) return a
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(a[3], b[3], t)]
}

export interface OverlayToggles {
  objInference: boolean
  showTrails: boolean
  drawCropBox: boolean
  showDeadZone: boolean
  showCenterLine: boolean
  showHighlight: boolean
}

export interface DrawParams {
  ctx: CanvasRenderingContext2D
  canvasW: number
  canvasH: number
  sourceW: number
  sourceH: number
  ms: number
  detections: LiveSample[]
  detOffsets: number[]
  plan: CropPlan | null
  toggles: OverlayToggles
  deadZoneWidth: number
}

function drawMarker(
  ctx: CanvasRenderingContext2D,
  bbox: [number, number, number, number] | null,
  color: string,
  label: string,
  sx: number,
  sy: number,
) {
  if (!bbox) return
  const [x, y, w] = bbox
  const cx = (x + w / 2) * sx
  const top = y * sy
  const size = 12
  const gap = 6
  const apexY = Math.max(0, top - gap)
  const baseY = Math.max(0, apexY - size)
  ctx.beginPath()
  ctx.moveTo(cx, apexY)
  ctx.lineTo(cx - size, baseY)
  ctx.lineTo(cx + size, baseY)
  ctx.closePath()
  ctx.fillStyle = color
  ctx.fill()
  ctx.lineWidth = 1
  ctx.strokeStyle = 'rgba(0,0,0,0.7)'
  ctx.stroke()
  ctx.font = '600 12px system-ui, sans-serif'
  ctx.lineWidth = 3
  ctx.strokeStyle = 'rgba(0,0,0,0.85)'
  ctx.strokeText(label, cx - 20, Math.max(12, baseY - 4))
  ctx.fillStyle = color
  ctx.fillText(label, cx - 20, Math.max(12, baseY - 4))
}

const TRAIL_MS = 2000 // 궤적 길이 (과거 2초)

/** 이동 궤적 — 최근 TRAIL_MS 동안의 중심점 폴리라인 (뒤로 갈수록 옅게). */
function drawTrails(p: DrawParams, sx: number, sy: number): void {
  const { ctx, ms, plan } = p
  const iNow = stepIndexAt(p.detOffsets, ms)
  if (iNow < 0) return
  const tMin = ms - TRAIL_MS

  const paint = (pts: [number, number][], color: string, width: number) => {
    for (let k = 1; k < pts.length; k++) {
      ctx.globalAlpha = 0.15 + 0.6 * (k / pts.length)
      ctx.lineWidth = width
      ctx.strokeStyle = color
      ctx.beginPath()
      ctx.moveTo(pts[k - 1][0], pts[k - 1][1])
      ctx.lineTo(pts[k][0], pts[k][1])
      ctx.stroke()
    }
    ctx.globalAlpha = 1
  }

  // 선수(track_id) 궤적 — 샘플 창에서 id별 중심점 수집
  const paths = new Map<number, [number, number][]>()
  for (let j = Math.max(0, iNow - 40); j <= iNow; j++) {
    if (p.detOffsets[j] < tMin) continue
    for (const d of p.detections[j].detections) {
      if (d.track_id == null || d.object_type !== 'player') continue
      const arr = paths.get(d.track_id) ?? []
      arr.push([(d.bbox_x + d.bbox_width / 2) * sx, (d.bbox_y + d.bbox_height / 2) * sy])
      paths.set(d.track_id, arr)
    }
  }
  for (const [id, pts] of paths) if (pts.length > 1) paint(pts, trackColor(id), 1.5)

  // 공 궤적 — 선정 트랙(plan.debug, 실측+보간)이 있으면 그걸로 (오탐 없는 깨끗한 선)
  if (plan?.debug?.length) {
    const pts: [number, number][] = []
    for (const e of plan.debug) {
      if (e.video_offset_ms < tMin || e.video_offset_ms > ms || !e.ball_bbox) continue
      const [x, y, w, h] = e.ball_bbox
      pts.push([(x + w / 2) * sx, (y + h / 2) * sy])
    }
    if (pts.length > 1) paint(pts, BALL_HL, 2.5)
  }
}

/** Draw all enabled overlays for time `ms` onto the canvas (cleared by the caller). */
export function drawOverlay(p: DrawParams): void {
  const { ctx, canvasW, canvasH, sourceW, sourceH, ms, plan, toggles } = p
  if (!sourceW || !sourceH) return
  const sx = canvasW / sourceW
  const sy = canvasH / sourceH

  if (toggles.showTrails && p.detections.length) drawTrails(p, sx, sy)

  // --- object inference: ByteTrack boxes, interpolated between samples ---
  if (toggles.objInference && p.detections.length) {
    ctx.lineWidth = 2
    ctx.font = '600 12px system-ui, sans-serif'
    for (const d of interpDetections(p.detections, p.detOffsets, ms)) {
      const color = trackColor(d.track_id ?? 0)
      const x = d.bbox_x * sx
      const y = d.bbox_y * sy
      const w = d.bbox_width * sx
      const h = d.bbox_height * sy
      ctx.lineWidth = 2
      ctx.strokeStyle = color
      ctx.strokeRect(x, y, w, h)
      const tag =
        (d.track_id != null ? `#${d.track_id} ` : '') +
        `${d.object_type} ${Math.round(d.confidence * 100)}%`
      ctx.lineWidth = 3
      ctx.strokeStyle = 'rgba(0,0,0,0.85)'
      ctx.strokeText(tag, x, Math.max(11, y - 4))
      ctx.fillStyle = color
      ctx.fillText(tag, x, Math.max(11, y - 4))
    }
  }

  if (!plan) return

  // --- crop trajectory geometry (interpolated → smooth) ---
  const cw = cropWidthFor(sourceH, sourceW)
  const cx = interpCenter(plan.samples, ms, sourceW)
  const left = leftEdge(cx, cw, sourceW)

  if (toggles.drawCropBox) {
    ctx.lineWidth = 2
    ctx.strokeStyle = CROP_COLOR
    ctx.strokeRect(left * sx, 0, cw * sx, sourceH * sy)
    ctx.font = '700 13px system-ui, sans-serif'
    ctx.fillStyle = CROP_COLOR
    ctx.fillText('CROP', left * sx + 6, 18)
  }

  // dead-zone band — ±half around the crop centre (crop is held when the ball is inside)
  if (toggles.showDeadZone && p.deadZoneWidth > 0) {
    const half = p.deadZoneWidth / 2
    ctx.lineWidth = 1
    ctx.strokeStyle = DEADZONE_COLOR
    for (const bx of [cx - half, cx + half]) {
      if (bx >= 0 && bx < sourceW) {
        ctx.beginPath()
        ctx.moveTo(bx * sx, 0)
        ctx.lineTo(bx * sx, canvasH)
        ctx.stroke()
      }
    }
  }

  if (toggles.showCenterLine) {
    ctx.lineWidth = 1.5
    ctx.strokeStyle = TARGET_COLOR
    ctx.beginPath()
    ctx.moveTo(cx * sx, 0)
    ctx.lineTo(cx * sx, canvasH)
    ctx.stroke()

    const tIdx = stepIndexAt(
      plan.samples.map((s) => s.video_offset_ms),
      ms,
    )
    const type = tIdx >= 0 ? plan.samples[tIdx].target_type : null
    if (type) {
      const text = `TARGET: ${type}`
      ctx.font = '700 13px system-ui, sans-serif'
      ctx.lineWidth = 4
      ctx.strokeStyle = 'rgba(0,0,0,0.85)'
      ctx.strokeText(text, 10, canvasH - 12)
      ctx.fillStyle = TYPE_COLORS[type] ?? '#dcdcdc'
      ctx.fillText(text, 10, canvasH - 12)
    }
  }

  // target highlight — the ball / carrier the pipeline actually selected (interpolated)
  if (toggles.showHighlight && plan.debug.length) {
    const offs = plan.debug.map((d) => d.video_offset_ms)
    const i = stepIndexAt(offs, ms)
    if (i >= 0) {
      const a = plan.debug[i]
      const b = i < plan.debug.length - 1 ? plan.debug[i + 1] : null
      const t = b && offs[i + 1] > offs[i] ? (ms - offs[i]) / (offs[i + 1] - offs[i]) : 0
      drawMarker(ctx, lerpBbox(a.ball_bbox, b?.ball_bbox ?? null, t), BALL_HL, 'BALL', sx, sy)
      drawMarker(ctx, lerpBbox(a.carrier_bbox, b?.carrier_bbox ?? null, t), CARRIER_HL, 'CARRIER', sx, sy)
    }
  }
}

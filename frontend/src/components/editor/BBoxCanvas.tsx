import { useEffect, useRef, useState } from 'react'
import { Image as KonvaImage, Layer, Rect, Stage, Transformer } from 'react-konva'
import type Konva from 'konva'
import {
  classColor,
  classColorAlpha,
  newBoxId,
  useEditorStore,
  type EditorBox,
} from '../../stores/editorStore'

interface Props {
  imageUrl: string
  width: number // container size
  height: number
}

interface Fit {
  scale: number
  x: number
  y: number
  iw: number
  ih: number
}

export default function BBoxCanvas({ imageUrl, width, height }: Props) {
  const { boxes, selectedId, tool, activeCls, select, updateBox, addBox } = useEditorStore()
  const drawMode = tool === 'draw'
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [fit, setFit] = useState<Fit | null>(null)
  const [temp, setTemp] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const stageRef = useRef<Konva.Stage>(null)
  const trRef = useRef<Konva.Transformer>(null)
  const rectRefs = useRef<Map<string, Konva.Rect>>(new Map())

  useEffect(() => {
    const img = new window.Image()
    img.src = imageUrl
    img.onload = () => setImage(img)
    return () => {
      img.onload = null
    }
  }, [imageUrl])

  useEffect(() => {
    if (!image) return
    const scale = Math.min(width / image.width, height / image.height)
    setFit({
      scale,
      x: (width - image.width * scale) / 2,
      y: (height - image.height * scale) / 2,
      iw: image.width,
      ih: image.height,
    })
    // reset zoom/pan on new image
    const stage = stageRef.current
    if (stage) {
      stage.scale({ x: 1, y: 1 })
      stage.position({ x: 0, y: 0 })
    }
  }, [image, width, height])

  // attach transformer to the selected rect (select tool only)
  useEffect(() => {
    const tr = trRef.current
    if (!tr) return
    const node = selectedId && !drawMode ? rectRefs.current.get(selectedId) : null
    tr.nodes(node ? [node] : [])
    tr.getLayer()?.batchDraw()
  }, [selectedId, boxes, drawMode])

  if (!image || !fit) {
    return <div style={{ width, height }} />
  }

  const toPx = (b: EditorBox) => ({
    x: fit.x + b.xyxy_n[0] * fit.iw * fit.scale,
    y: fit.y + b.xyxy_n[1] * fit.ih * fit.scale,
    w: (b.xyxy_n[2] - b.xyxy_n[0]) * fit.iw * fit.scale,
    h: (b.xyxy_n[3] - b.xyxy_n[1]) * fit.ih * fit.scale,
  })

  const toNorm = (x: number, y: number, w: number, h: number): [number, number, number, number] => {
    const clamp = (v: number) => Math.min(1, Math.max(0, v))
    return [
      clamp((x - fit.x) / (fit.iw * fit.scale)),
      clamp((y - fit.y) / (fit.ih * fit.scale)),
      clamp((x + w - fit.x) / (fit.iw * fit.scale)),
      clamp((y + h - fit.y) / (fit.ih * fit.scale)),
    ]
  }

  const stagePointer = (): { x: number; y: number } | null => {
    const stage = stageRef.current
    if (!stage) return null
    const pos = stage.getPointerPosition()
    if (!pos) return null
    // convert from viewport coords into stage-content coords (zoom/pan aware)
    const scale = stage.scaleX()
    return { x: (pos.x - stage.x()) / scale, y: (pos.y - stage.y()) / scale }
  }

  // clamp stage position so the image always covers the viewport — panning is
  // confined to within the image, never revealing empty space around it. When an
  // axis is smaller than the viewport (not zoomed past fit) the image is centered.
  const clampPos = (pos: { x: number; y: number }): { x: number; y: number } => {
    const stage = stageRef.current
    if (!stage) return pos
    const s = stage.scaleX()
    const imgW = fit.iw * fit.scale
    const imgH = fit.ih * fit.scale
    const maxX = -fit.x * s
    const minX = width - (fit.x + imgW) * s
    const maxY = -fit.y * s
    const minY = height - (fit.y + imgH) * s
    const clamp = (v: number, lo: number, hi: number) =>
      lo <= hi ? Math.max(lo, Math.min(hi, v)) : (lo + hi) / 2
    return { x: clamp(pos.x, minX, maxX), y: clamp(pos.y, minY, maxY) }
  }

  const onWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()
    const stage = stageRef.current
    if (!stage) return
    const oldScale = stage.scaleX()
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    const factor = e.evt.deltaY > 0 ? 1 / 1.08 : 1.08
    // floor at 1 (fitted view): never zoom out smaller than the image fit
    const newScale = Math.min(8, Math.max(1, oldScale * factor))
    const mousePoint = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    }
    stage.scale({ x: newScale, y: newScale })
    stage.position(
      clampPos({ x: pointer.x - mousePoint.x * newScale, y: pointer.y - mousePoint.y * newScale }),
    )
  }

  const onStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // draw tool: boxes don't listen, so any press starts a new box
    if (drawMode) {
      const p = stagePointer()
      if (p) setTemp({ x0: p.x, y0: p.y, x1: p.x, y1: p.y })
      return
    }
    const clickedEmpty = e.target === e.target.getStage() || e.target.name() === 'bg-image'
    if (clickedEmpty) select(null)
  }

  const onStageMouseMove = () => {
    if (!temp) return
    const p = stagePointer()
    if (p) setTemp({ ...temp, x1: p.x, y1: p.y })
  }

  const onStageMouseUp = () => {
    if (!temp) return
    const x = Math.min(temp.x0, temp.x1)
    const y = Math.min(temp.y0, temp.y1)
    const w = Math.abs(temp.x1 - temp.x0)
    const h = Math.abs(temp.y1 - temp.y0)
    setTemp(null)
    if (w > 5 && h > 5) {
      // stay in draw tool so consecutive boxes can be drawn without re-toggling
      addBox({ id: newBoxId(), cls: activeCls, xyxy_n: toNorm(x, y, w, h), status: 'edited' })
    }
  }

  return (
    <Stage
      ref={stageRef}
      width={width}
      height={height}
      draggable={!drawMode && !temp}
      dragBoundFunc={clampPos}
      onWheel={onWheel}
      onMouseDown={onStageMouseDown}
      onMouseMove={onStageMouseMove}
      onMouseUp={onStageMouseUp}
      style={{ cursor: drawMode ? 'crosshair' : 'default', background: '#111' }}
    >
      <Layer>
        <KonvaImage
          name="bg-image"
          image={image}
          x={fit.x}
          y={fit.y}
          scaleX={fit.scale}
          scaleY={fit.scale}
        />
        {boxes.map((b) => {
          const px = toPx(b)
          const color = classColor(b.cls)
          const flagged = b.status === 'needs_review'
          const selected = b.id === selectedId
          return (
            <Rect
              key={b.id}
              ref={(node) => {
                if (node) rectRefs.current.set(b.id, node)
                else rectRefs.current.delete(b.id)
              }}
              x={px.x}
              y={px.y}
              width={px.w}
              height={px.h}
              stroke={color}
              strokeWidth={selected ? 3 : 2}
              strokeScaleEnabled={false}
              dash={flagged ? [8, 4] : undefined}
              fill={selected ? classColorAlpha(b.cls, 0.16) : 'transparent'}
              listening={!drawMode}
              draggable={!drawMode}
              onClick={() => select(b.id)}
              onTap={() => select(b.id)}
              onDragEnd={(e) => {
                const node = e.target
                updateBox(b.id, { xyxy_n: toNorm(node.x(), node.y(), node.width(), node.height()) })
              }}
              onTransformEnd={(e) => {
                const node = e.target as Konva.Rect
                const w = Math.max(4, node.width() * node.scaleX())
                const h = Math.max(4, node.height() * node.scaleY())
                node.scaleX(1)
                node.scaleY(1)
                updateBox(b.id, { xyxy_n: toNorm(node.x(), node.y(), w, h) })
              }}
            />
          )
        })}
        {temp && (
          <Rect
            x={Math.min(temp.x0, temp.x1)}
            y={Math.min(temp.y0, temp.y1)}
            width={Math.abs(temp.x1 - temp.x0)}
            height={Math.abs(temp.y1 - temp.y0)}
            stroke={classColor(activeCls)}
            strokeWidth={2}
            strokeScaleEnabled={false}
            dash={[4, 4]}
          />
        )}
        <Transformer
          ref={trRef}
          rotateEnabled={false}
          flipEnabled={false}
          ignoreStroke
          boundBoxFunc={(oldBox, newBox) => (newBox.width < 4 || newBox.height < 4 ? oldBox : newBox)}
        />
      </Layer>
    </Stage>
  )
}

import type { PredictBox } from '../../api/client'

// Class → color. Stable by class id so the same class keeps its color across runs.
const PALETTE = [
  '#4dabf7', '#51cf66', '#ff6b6b', '#ffd43b', '#cc5de8',
  '#20c997', '#ff922b', '#f783ac', '#748ffc', '#a9e34b',
]
export const clsColor = (cls: number) => PALETTE[((cls % PALETTE.length) + PALETTE.length) % PALETTE.length]

export const boxKey = (b: PredictBox) => `${b.cls}:${b.xyxyn.join(',')}`

interface Props {
  src: string
  boxes: PredictBox[] // already filtered (conf + class visibility)
  hovered?: string | null
  onHover?: (key: string | null) => void
  showLabels?: boolean
}

/** Read-only detection overlay: absolutely-positioned boxes (percentage coords)
 * over the image, so it stays pixel-accurate at any responsive width. */
export default function BoxOverlay({ src, boxes, hovered, onHover, showLabels = true }: Props) {
  return (
    <div style={{ position: 'relative', width: '100%', lineHeight: 0 }}>
      <img src={src} alt="preview" style={{ width: '100%', display: 'block', borderRadius: 8 }} />
      {boxes.map((b) => {
        const [x1, y1, x2, y2] = b.xyxyn
        const key = boxKey(b)
        const color = clsColor(b.cls)
        const active = hovered === key
        return (
          <div
            key={key}
            onMouseEnter={() => onHover?.(key)}
            onMouseLeave={() => onHover?.(null)}
            style={{
              position: 'absolute',
              left: `${x1 * 100}%`,
              top: `${y1 * 100}%`,
              width: `${(x2 - x1) * 100}%`,
              height: `${(y2 - y1) * 100}%`,
              border: `2px solid ${color}`,
              background: active ? `${color}22` : 'transparent',
              boxShadow: active ? `0 0 0 2px ${color}` : 'none',
              borderRadius: 2,
              pointerEvents: 'auto',
              cursor: 'default',
              zIndex: active ? 2 : 1,
            }}
          >
            {showLabels && (
              <span
                style={{
                  position: 'absolute',
                  top: -17,
                  left: -2,
                  background: color,
                  color: '#000',
                  fontSize: 11,
                  fontWeight: 600,
                  padding: '0 4px',
                  borderRadius: 3,
                  whiteSpace: 'nowrap',
                  lineHeight: '16px',
                }}
              >
                {b.name} {(b.score * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}

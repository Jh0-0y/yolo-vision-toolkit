import { Group, NumberInput, Select, Text } from '@mantine/core'
import type { ModelOut } from '../../api/client'
import { IMGSZ_OPTIONS } from './useAnnotateJob'

/** 공 전용 검출 설정 — model이 null이면 "Same as player" (단일 모델 모드). */
export interface BallDetector {
  modelId: string | null
  conf: number | ''
  tileSize: number
  stride: number
  mergeIou: number
}

export const DEFAULT_BALL: BallDetector = {
  modelId: null,
  conf: '',
  tileSize: 640,
  stride: 480,
  mergeIou: 0.5,
}

interface Props {
  models: ModelOut[]
  modelId: string | null
  onModelId: (v: string | null) => void
  imgsz: number
  onImgsz: (v: number) => void
  conf: number | ''
  onConf: (v: number | '') => void
  sampling: number | ''
  onSampling: (v: number | '') => void
  ball: BallDetector
  onBall: (v: BallDetector) => void
  disabled?: boolean
  description?: string
}

const SAME_AS_PLAYER = '__same__'

/** Shared detection settings — identical UI for the Crop JSON and Crop Draw tabs.
 *  Player: full-frame track (ByteTrack). Ball: 전용 모델 선택 시 타일 추론(batch+NMS). */
export default function DetectionSettings({
  models, modelId, onModelId, imgsz, onImgsz, conf, onConf, sampling, onSampling,
  ball, onBall, disabled, description,
}: Props) {
  const setBall = (patch: Partial<BallDetector>) => onBall({ ...ball, ...patch })
  const overlap = ball.tileSize - ball.stride
  const split = ball.modelId != null

  return (
    <>
      <Text size="sm" fw={600}>Detection</Text>
      {description && (
        <Text size="xs" c="dimmed">{description}</Text>
      )}

      <Text size="xs" fw={600} c="dimmed">Player — full frame (ByteTrack)</Text>
      <Select
        label="Model"
        placeholder={models.length ? 'Pick a model' : 'No models — train or upload one first'}
        data={models.map((m) => ({ value: m.id, label: m.name }))}
        value={modelId}
        onChange={onModelId}
        disabled={disabled || !models.length}
        allowDeselect={false}
      />
      <Group grow align="flex-start">
        <Select
          label="Image size"
          description="Match your model's training size"
          data={IMGSZ_OPTIONS}
          value={String(imgsz)}
          onChange={(v) => v && onImgsz(Number(v))}
          disabled={disabled}
          allowDeselect={false}
        />
        <NumberInput
          label="Confidence"
          description="Detection threshold"
          placeholder="default 0.1"
          value={conf}
          onChange={(v) => onConf(v === '' || v == null ? '' : Number(v))}
          min={0.05}
          max={0.95}
          step={0.05}
          decimalScale={2}
          disabled={disabled}
        />
        <NumberInput
          label="Sampling interval (ms)"
          description="Lower = smoother, slower"
          placeholder="default 100"
          value={sampling}
          onChange={(v) => onSampling(v === '' || v == null ? '' : Number(v))}
          min={10}
          step={10}
          disabled={disabled}
        />
      </Group>

      <Text size="xs" fw={600} c="dimmed" mt={4}>
        Ball — {split ? 'tiled inference (batch + NMS merge)' : 'same model as player'}
      </Text>
      <Group grow align="flex-start">
        <Select
          label="Ball model"
          description="Pick a tile-trained ball model to enable tiled inference"
          data={[
            { value: SAME_AS_PLAYER, label: 'Same as player' },
            ...models.map((m) => ({ value: m.id, label: m.name })),
          ]}
          value={ball.modelId ?? SAME_AS_PLAYER}
          onChange={(v) => setBall({ modelId: v === SAME_AS_PLAYER ? null : v })}
          disabled={disabled || !models.length}
          allowDeselect={false}
        />
        {split && (
          <NumberInput
            label="Ball confidence"
            description="Defaults to player conf"
            placeholder="default 0.1"
            value={ball.conf}
            onChange={(v) => setBall({ conf: v === '' || v == null ? '' : Number(v) })}
            min={0.05}
            max={0.95}
            step={0.05}
            decimalScale={2}
            disabled={disabled}
          />
        )}
      </Group>
      {split && (
        <>
          <Group grow align="flex-start">
            <NumberInput
              label="Tile size (px)"
              description="Match the training tile size"
              value={ball.tileSize}
              onChange={(v) => setBall({ tileSize: Number(v) || 640 })}
              min={64}
              step={32}
              disabled={disabled}
            />
            <NumberInput
              label="Stride (px)"
              description={`Overlap = ${overlap}px${overlap <= 0 ? ' — needs overlap!' : ''}`}
              value={ball.stride}
              onChange={(v) => setBall({ stride: Number(v) || 480 })}
              min={32}
              step={32}
              max={ball.tileSize}
              disabled={disabled}
              error={overlap <= 0 ? 'Stride must be < tile size' : undefined}
            />
            <NumberInput
              label="Merge IoU"
              description="NMS for duplicate boxes in overlaps"
              value={ball.mergeIou}
              onChange={(v) => setBall({ mergeIou: v === '' || v == null ? 0.5 : Number(v) })}
              min={0.1}
              max={0.9}
              step={0.05}
              decimalScale={2}
              disabled={disabled}
            />
          </Group>
          <Text size="xs" c="dimmed">
            1920×1080 with 640/480 → 8 tiles per frame, batched in one GPU call.
            Ball boxes carry no track id — the clip planner stitches the trajectory.
          </Text>
        </>
      )}
    </>
  )
}

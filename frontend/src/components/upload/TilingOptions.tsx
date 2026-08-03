import { Checkbox, Group, NumberInput, Stack, Switch, Text } from '@mantine/core'

export interface TilingState {
  tile: boolean
  tileSize: number
  stride: number
  minVisibility: number
  dropEmpty: boolean
}

export const DEFAULT_TILING: TilingState = {
  tile: false,
  tileSize: 640,
  stride: 480,
  minVisibility: 0.6, // 데이터 특성상 60% 이상 보여야 탐지 (프로젝트 기본)
  dropEmpty: true,
}

interface Props {
  value: TilingState
  onChange: (v: TilingState) => void
  disabled?: boolean
  /** 라벨 zip 임포트에서만 true — min_visibility/drop_empty 노출 */
  withLabels?: boolean
}

/** 타일링 옵션 — 1920px 원본을 학습용 타일로 쪼갠다 (1920×1080 + 640/480 = 8타일/장). */
export default function TilingOptions({ value, onChange, disabled, withLabels }: Props) {
  const overlap = value.tileSize - value.stride
  const set = (patch: Partial<TilingState>) => onChange({ ...value, ...patch })

  return (
    <Stack gap="sm">
      <Switch
        label="Tile into training patches"
        description="Split each 1920px frame into overlapping tiles so small objects train larger"
        checked={value.tile}
        onChange={(e) => set({ tile: e.currentTarget.checked })}
        disabled={disabled}
      />
      {value.tile && (
        <>
          <Group grow align="flex-start">
            <NumberInput
              label="Tile size (px)"
              description="Square tile edge"
              value={value.tileSize}
              onChange={(v) => set({ tileSize: Number(v) || 640 })}
              min={64}
              step={32}
              disabled={disabled}
            />
            <NumberInput
              label="Stride (px)"
              description={`Overlap = ${overlap}px${overlap <= 0 ? ' — needs overlap!' : ''}`}
              value={value.stride}
              onChange={(v) => set({ stride: Number(v) || 480 })}
              min={32}
              step={32}
              max={value.tileSize}
              disabled={disabled}
              error={overlap <= 0 ? 'Stride must be < tile size — boundary objects split in two' : undefined}
            />
          </Group>
          <Text size="xs" c="dimmed">
            1920×1080 with 640/480 → 4×2 = 8 tiles per frame. Overlap must exceed the
            largest object so it lands whole in some tile; the last tile clamps to the frame edge.
          </Text>
          {withLabels && (
            <>
              <NumberInput
                label="Min visibility (0–1)"
                description="Boxes visible at least this fraction are clipped & kept; below are dropped"
                value={value.minVisibility}
                onChange={(v) => set({ minVisibility: v === '' || v == null ? 0.6 : Number(v) })}
                min={0}
                max={1}
                step={0.05}
                decimalScale={2}
                disabled={disabled}
              />
              <Checkbox
                label="Drop tiles with no remaining boxes"
                checked={value.dropEmpty}
                onChange={(e) => set({ dropEmpty: e.currentTarget.checked })}
                disabled={disabled}
                size="xs"
              />
            </>
          )}
        </>
      )}
    </Stack>
  )
}

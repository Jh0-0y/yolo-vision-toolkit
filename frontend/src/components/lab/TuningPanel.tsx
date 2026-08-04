import { useState } from 'react'
import {
  Button,
  Checkbox,
  Group,
  NumberInput,
  SimpleGrid,
  Stack,
  Text,
  UnstyledButton,
} from '@mantine/core'
import { IconChevronRight } from '@tabler/icons-react'
import type { TrackcropOverrides } from '../../api/client'

type NumberKnobKey = Exclude<keyof TrackcropOverrides, 'use_carrier'>

interface Knob {
  key: NumberKnobKey
  label: string
  def: number
  step?: number
  min?: number
  max?: number
}

// Core — directly affects crop behaviour
const CORE: Knob[] = [
  { key: 'dead_zone_width', label: 'Dead zone width (px)', def: 208, step: 8, min: 0 },
  { key: 'sampling_interval_ms', label: 'Sampling interval (ms)', def: 100, step: 10, min: 10 },
  { key: 'max_move_px_per_second', label: 'Max move speed (px/s)', def: 1200, step: 100, min: 0 },
  { key: 'ball_weight', label: 'Ball weight (0–1)', def: 0.7, step: 0.05, min: 0, max: 1 },
  { key: 'interp_max_gap_ms', label: 'Interpolate gap ≤ (ms)', def: 2000, step: 100, min: 0 },
  { key: 'stitch_max_gap_ms', label: 'Stitch gap ≤ (ms)', def: 2500, step: 100, min: 0 },
  { key: 'min_track_score', label: 'Min track score (0–1)', def: 0.25, step: 0.05, min: 0, max: 1 },
]

// Advanced — track stitching/scoring & path-optimization internals
const ADVANCED: Knob[] = [
  { key: 'ball_max_speed_px_s', label: 'Ball max speed (px/s)', def: 3000, step: 100, min: 0 },
  { key: 'split_base_px', label: 'Split: base jump (px)', def: 80, step: 10, min: 0 },
  { key: 'stitch_base_px', label: 'Stitch: base dist (px)', def: 150, step: 10, min: 0 },
  { key: 'stitch_velocity_cap_ms', label: 'Stitch: extrapolate cap (ms)', def: 500, step: 50, min: 0 },
  { key: 'w_travel', label: 'Score: travel weight', def: 0.4, step: 0.05, min: 0, max: 1 },
  { key: 'w_interaction', label: 'Score: interaction weight', def: 0.4, step: 0.05, min: 0, max: 1 },
  { key: 'w_span', label: 'Score: span weight', def: 0.2, step: 0.05, min: 0, max: 1 },
  { key: 'travel_norm_px', label: 'Score: travel norm (px)', def: 1920, step: 100, min: 1 },
  { key: 'absorb_allow_scale', label: 'Absorb-merge scale', def: 2, step: 0.5, min: 1 },
  { key: 'prune_dev_px', label: 'Prune deviation (px)', def: 200, step: 25, min: 50 },
  { key: 'possession_margin', label: 'Possession margin (×height)', def: 0.15, step: 0.05, min: 0 },
  { key: 'w_follow', label: 'Path: follow weight', def: 1, step: 0.1, min: 0 },
  { key: 'w_inside', label: 'Path: inside pull', def: 0.05, step: 0.01, min: 0 },
  { key: 'w_vel', label: 'Path: velocity penalty', def: 0.1, step: 0.05, min: 0 },
  { key: 'w_acc', label: 'Path: accel penalty', def: 15, step: 5, min: 0 },
  { key: 'irls_iters', label: 'Path: solver iters', def: 30, step: 5, min: 1 },
  { key: 'min_follow_conf', label: 'Path: min follow conf', def: 0.1, step: 0.05, min: 0, max: 1 },
]

interface Props {
  value: TrackcropOverrides
  onChange: (v: TrackcropOverrides) => void
  disabled?: boolean
  /** knobs to hide — e.g. live preview hides sampling_interval_ms (a detection param). */
  exclude?: (keyof TrackcropOverrides)[]
}

/** trackcrop clip-planner tuning knobs — leaving a field empty omits it, so the default is used. */
export default function TuningPanel({ value, onChange, disabled, exclude }: Props) {
  const [showTuning, setShowTuning] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const hidden = new Set(exclude ?? [])
  const core = CORE.filter((k) => !hidden.has(k.key))
  const advanced = ADVANCED.filter((k) => !hidden.has(k.key))

  const set = (k: NumberKnobKey, v: number | string) => {
    const next = { ...value }
    if (v === '' || v == null || Number.isNaN(Number(v))) delete next[k]
    else next[k] = Number(v)
    onChange(next)
  }

  const setCarrier = (checked: boolean) => {
    const next = { ...value }
    if (checked) next.use_carrier = true
    else delete next.use_carrier
    onChange(next)
  }

  const renderKnob = (kn: Knob) => (
    <NumberInput
      key={kn.key}
      label={kn.label}
      placeholder={`default ${kn.def}`}
      value={value[kn.key] ?? ''}
      onChange={(v) => set(kn.key, v)}
      step={kn.step}
      min={kn.min}
      max={kn.max}
      disabled={disabled}
      size="xs"
    />
  )

  const count = Object.keys(value).length

  return (
    <Stack gap="xs">
      <Group justify="space-between">
        <UnstyledButton onClick={() => setShowTuning((v) => !v)} disabled={disabled}>
          <Group gap={4}>
            <IconChevronRight
              size={14}
              style={{
                transform: showTuning ? 'rotate(90deg)' : 'none',
                transition: 'transform 150ms',
              }}
            />
            <Text size="sm" fw={600}>
              Crop {count > 0 && <Text span c="dimmed" size="xs">({count} overrides)</Text>}
            </Text>
          </Group>
        </UnstyledButton>
        {count > 0 && (
          <Button size="compact-xs" variant="subtle" onClick={() => onChange({})} disabled={disabled}>
            Reset
          </Button>
        )}
      </Group>

      {showTuning && (
        <>
          <SimpleGrid cols={{ base: 2, sm: 3 }} spacing="xs">
            {core.map(renderKnob)}
          </SimpleGrid>
          <Checkbox
            size="xs"
            label="Track last ball carrier during long occlusion (use_carrier)"
            checked={value.use_carrier ?? false}
            onChange={(e) => setCarrier(e.currentTarget.checked)}
            disabled={disabled}
          />
        </>
      )}

      <UnstyledButton onClick={() => setShowAdvanced((v) => !v)} disabled={disabled}>
        <Group gap={4}>
          <IconChevronRight
            size={14}
            style={{
              transform: showAdvanced ? 'rotate(90deg)' : 'none',
              transition: 'transform 150ms',
            }}
          />
          <Text size="sm" fw={600}>Advanced</Text>
        </Group>
      </UnstyledButton>
      {showAdvanced && (
        <SimpleGrid cols={{ base: 2, sm: 3 }} spacing="xs">
          {advanced.map(renderKnob)}
        </SimpleGrid>
      )}
    </Stack>
  )
}

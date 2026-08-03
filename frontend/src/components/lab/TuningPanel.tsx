import { useState } from 'react'
import {
  Button,
  Group,
  NumberInput,
  SimpleGrid,
  Stack,
  Text,
  UnstyledButton,
} from '@mantine/core'
import { IconChevronRight } from '@tabler/icons-react'
import type { TrackcropOverrides } from '../../api/client'

interface Knob {
  key: keyof TrackcropOverrides
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
  { key: 'ball_lost_hold_ms', label: 'Ball lost hold (ms)', def: 1000, step: 100, min: 0 },
  { key: 'player_lost_hold_ms', label: 'Player lost hold (ms)', def: 1500, step: 100, min: 0 },
  { key: 'max_move_px_per_second', label: 'Max move speed (px/s)', def: 1200, step: 100, min: 0 },
  { key: 'ball_weight', label: 'Ball weight (0–1)', def: 0.7, step: 0.05, min: 0, max: 1 },
  { key: 'match_max_jump_px', label: 'Ball gate ± base (px, 0=off)', def: 300, step: 20, min: 0 },
]

// Advanced — stabilization & selection internals
const ADVANCED: Knob[] = [
  { key: 'outlier_deviation_px', label: 'Outlier deviation (px)', def: 300, step: 10, min: 0 },
  { key: 'outlier_window', label: 'Outlier window', def: 5, step: 2, min: 1 },
  { key: 'smooth_window', label: 'Smooth window', def: 5, step: 2, min: 1 },
  { key: 'low_confidence', label: 'Low-confidence threshold', def: 0.3, step: 0.05, min: 0, max: 1 },
  { key: 'confidence_decay', label: 'Confidence decay', def: 0.9, step: 0.05, min: 0, max: 1 },
  { key: 'reacquire_blend', label: 'Reacquire blend', def: 0.5, step: 0.05, min: 0, max: 1 },
  { key: 'w_continuity', label: 'Ball: continuity weight', def: 0.5, step: 0.05, min: 0, max: 1 },
  { key: 'w_confidence', label: 'Ball: conf weight', def: 0.3, step: 0.05, min: 0, max: 1 },
  { key: 'w_player_proximity', label: 'Ball: player-proximity weight', def: 0.2, step: 0.05, min: 0, max: 1 },
  { key: 'w_carrier_proximity', label: 'Carrier: proximity weight', def: 0.6, step: 0.05, min: 0, max: 1 },
  { key: 'w_carrier_velocity', label: 'Carrier: velocity weight', def: 0.4, step: 0.05, min: 0, max: 1 },
  { key: 'velocity_scale', label: 'Velocity scale (px/ms)', def: 0.5, step: 0.1, min: 0.01 },
]

interface Props {
  value: TrackcropOverrides
  onChange: (v: TrackcropOverrides) => void
  disabled?: boolean
  /** knobs to hide — e.g. live preview hides sampling_interval_ms (a detection param). */
  exclude?: (keyof TrackcropOverrides)[]
}

/** trackcrop tuning knobs — leaving a field empty omits it, so the default (constants) is used. */
export default function TuningPanel({ value, onChange, disabled, exclude }: Props) {
  const [showTuning, setShowTuning] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const hidden = new Set(exclude ?? [])
  const core = CORE.filter((k) => !hidden.has(k.key))
  const advanced = ADVANCED.filter((k) => !hidden.has(k.key))

  const set = (k: keyof TrackcropOverrides, v: number | string) => {
    const next = { ...value }
    if (v === '' || v == null || Number.isNaN(Number(v))) delete next[k]
    else next[k] = Number(v)
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
        <SimpleGrid cols={{ base: 2, sm: 3 }} spacing="xs">
          {core.map(renderKnob)}
        </SimpleGrid>
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

      <Text size="xs" c="dimmed">
        Empty fields use the default value.
      </Text>
    </Stack>
  )
}

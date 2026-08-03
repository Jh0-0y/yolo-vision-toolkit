import { Group, NumberInput, Select, Text } from '@mantine/core'
import type { ModelOut } from '../../api/client'
import { IMGSZ_OPTIONS } from './useAnnotateJob'

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
  disabled?: boolean
  description?: string
}

/** Shared detection settings — identical UI for the Crop JSON and Crop Draw tabs. */
export default function DetectionSettings({
  models, modelId, onModelId, imgsz, onImgsz, conf, onConf, sampling, onSampling,
  disabled, description,
}: Props) {
  return (
    <>
      <Text size="sm" fw={600}>Detection</Text>
      {description && (
        <Text size="xs" c="dimmed">{description}</Text>
      )}
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
    </>
  )
}

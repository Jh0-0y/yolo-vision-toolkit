import { ActionIcon, Button, Card, Group, NumberInput, SegmentedControl, Select, Stack, Text } from '@mantine/core'
import { IconPlus, IconX } from '@tabler/icons-react'
import type { ModelOut } from '../../api/client'

/** Full scan 의 추론 해상도 후보. 크게 잡을수록 작은 공이 살고 느려진다. */
export const IMGSZ_OPTIONS = ['640', '960', '1280', '1600', '1920']

/** 검출기 엔트리 — 모두 대등하다. 역할(추적/공)은 백엔드가 모드에서 자동 유도:
 *  첫 번째 Full scan 엔트리가 ByteTrack 추적, 나머지는 공 검출에 기여. */
export interface DetectorEntry {
  key: string
  modelId: string | null
  mode: 'full' | 'tiled'
  conf: number | ''
  imgsz: number // full 모드
  tileSize: number // tiled 모드
  stride: number
  mergeIou: number
}

let entrySeq = 0

export function newEntry(mode: 'full' | 'tiled' = 'full'): DetectorEntry {
  entrySeq += 1
  return {
    key: `det-${entrySeq}`,
    modelId: null,
    mode,
    conf: '',
    imgsz: 1280,
    tileSize: 640,
    stride: 480,
    mergeIou: 0.5,
  }
}

interface Props {
  models: ModelOut[]
  entries: DetectorEntry[]
  onEntries: (v: DetectorEntry[]) => void
  sampling: number | ''
  onSampling: (v: number | '') => void
  disabled?: boolean
}

/** Shared detection settings — 모델 엔트리 리스트.
 *  각 엔트리는 모델 + 추론 방식(Full scan / Tiled)과 방식별 옵션을 독립적으로 가진다. */
export default function DetectionSettings({
  models, entries, onEntries, sampling, onSampling, disabled,
}: Props) {
  const modelData = models.map((m) => ({ value: m.id, label: m.name }))

  const patch = (key: string, p: Partial<DetectorEntry>) =>
    onEntries(entries.map((e) => (e.key === key ? { ...e, ...p } : e)))
  const remove = (key: string) => onEntries(entries.filter((e) => e.key !== key))
  const add = () => {
    const e = newEntry('tiled')
    e.modelId = models[0]?.id ?? null
    onEntries([...entries, e])
  }

  return (
    <>
      <Group justify="space-between">
        <Text size="sm" fw={600}>Detection</Text>
        <Button
          size="compact-xs"
          variant="light"
          leftSection={<IconPlus size={14} />}
          onClick={add}
          disabled={disabled || !models.length}
        >
          Add model
        </Button>
      </Group>

      {entries.map((entry) => {
        const overlap = entry.tileSize - entry.stride
        return (
          <Card key={entry.key} withBorder radius="sm" padding="sm">
            <Stack gap="xs">
              <Group justify="space-between">
                <SegmentedControl
                  size="xs"
                  value={entry.mode}
                  onChange={(v) => patch(entry.key, { mode: v as 'full' | 'tiled' })}
                  data={[
                    { value: 'full', label: 'Full scan' },
                    { value: 'tiled', label: 'Tiled' },
                  ]}
                  disabled={disabled}
                />
                {entries.length > 1 && (
                  <ActionIcon
                    variant="subtle"
                    color="gray"
                    size="sm"
                    onClick={() => remove(entry.key)}
                    disabled={disabled}
                  >
                    <IconX size={14} />
                  </ActionIcon>
                )}
              </Group>

              <Group grow align="flex-start">
                <Select
                  label="Model"
                  placeholder={models.length ? 'Pick a model' : 'No models yet'}
                  data={modelData}
                  value={entry.modelId}
                  onChange={(v) => patch(entry.key, { modelId: v })}
                  disabled={disabled || !models.length}
                  allowDeselect={false}
                />
                <NumberInput
                  label="Confidence"
                  placeholder="default 0.1"
                  value={entry.conf}
                  onChange={(v) => patch(entry.key, { conf: v === '' || v == null ? '' : Number(v) })}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  decimalScale={2}
                  disabled={disabled}
                />
                {entry.mode === 'full' ? (
                  <Select
                    label="Image size"
                    data={IMGSZ_OPTIONS}
                    value={String(entry.imgsz)}
                    onChange={(v) => v && patch(entry.key, { imgsz: Number(v) })}
                    disabled={disabled}
                    allowDeselect={false}
                  />
                ) : (
                  <NumberInput
                    label="Tile size (px)"
                    value={entry.tileSize}
                    onChange={(v) => patch(entry.key, { tileSize: Number(v) || 640 })}
                    min={64}
                    step={32}
                    disabled={disabled}
                  />
                )}
              </Group>

              {entry.mode === 'tiled' && (
                <Group grow align="flex-start">
                  <NumberInput
                    label="Stride (px)"
                    value={entry.stride}
                    onChange={(v) => patch(entry.key, { stride: Number(v) || 480 })}
                    min={32}
                    step={32}
                    max={entry.tileSize}
                    disabled={disabled}
                    error={overlap <= 0 ? 'Stride must be < tile size' : undefined}
                  />
                  <NumberInput
                    label="Merge IoU"
                    value={entry.mergeIou}
                    onChange={(v) => patch(entry.key, { mergeIou: v === '' || v == null ? 0.5 : Number(v) })}
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    decimalScale={2}
                    disabled={disabled}
                  />
                </Group>
              )}
            </Stack>
          </Card>
        )
      })}

      <NumberInput
        label="Sampling interval (ms)"
        description="Lower = smoother, slower"
        placeholder="default 100"
        value={sampling}
        onChange={(v) => onSampling(v === '' || v == null ? '' : Number(v))}
        min={10}
        step={10}
        disabled={disabled}
        maw={260}
      />
    </>
  )
}

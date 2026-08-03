import { ActionIcon, Badge, Button, Card, Group, NumberInput, SegmentedControl, Select, Stack, Text } from '@mantine/core'
import { IconPlus, IconX } from '@tabler/icons-react'
import type { ModelOut } from '../../api/client'
import { IMGSZ_OPTIONS } from './useAnnotateJob'

/** 검출기 엔트리 — 첫 번째는 베이스(선수+ByteTrack), 추가 엔트리는 공 전담. */
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

/** Shared detection settings — 모델 리스트 방식.
 *  기본 1개(베이스: 풀 프레임 + ByteTrack). "Add model"로 공 전담 검출기를 추가하고
 *  엔트리마다 풀스캔/타일 추론을 고른다. 추가 검출기가 있으면 공은 그쪽이 전담한다. */
export default function DetectionSettings({
  models, entries, onEntries, sampling, onSampling, disabled,
}: Props) {
  const modelData = models.map((m) => ({ value: m.id, label: m.name }))
  const hasExtras = entries.length > 1

  const patch = (key: string, p: Partial<DetectorEntry>) =>
    onEntries(entries.map((e) => (e.key === key ? { ...e, ...p } : e)))
  const remove = (key: string) => onEntries(entries.filter((e) => e.key !== key))
  const add = () => {
    const e = newEntry('tiled') // 추가 모델의 주 용도 = 타일 학습된 공 모델
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

      {entries.map((entry, i) => {
        const base = i === 0
        const overlap = entry.tileSize - entry.stride
        return (
          <Card key={entry.key} withBorder radius="sm" padding="sm">
            <Stack gap="xs">
              <Group justify="space-between">
                <Group gap="xs">
                  <Badge variant="light" size="sm">
                    {base
                      ? hasExtras ? 'Players — ByteTrack' : 'Players + ball — ByteTrack'
                      : 'Ball'}
                  </Badge>
                  {!base && (
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
                  )}
                </Group>
                {!base && (
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
                  placeholder={base ? 'default 0.1' : 'same as base'}
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
                    description={base ? undefined : 'Full-frame inference size'}
                    data={IMGSZ_OPTIONS}
                    value={String(entry.imgsz)}
                    onChange={(v) => v && patch(entry.key, { imgsz: Number(v) })}
                    disabled={disabled}
                    allowDeselect={false}
                  />
                ) : (
                  <NumberInput
                    label="Tile size (px)"
                    description="Match the training tile"
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
                    description={`Overlap = ${overlap}px${overlap <= 0 ? ' — needs overlap!' : ''}`}
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
                    description="NMS for overlap duplicates"
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

      {hasExtras && (
        <Text size="xs" c="dimmed">
          Added models detect the ball only (no track id — the clip planner stitches the
          trajectory); the base model keeps the players.
        </Text>
      )}

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

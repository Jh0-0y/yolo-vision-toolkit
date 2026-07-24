import { Badge, Card, Checkbox, Group, Loader, SegmentedControl, Select, Slider, Stack, Switch, Text } from '@mantine/core'
import type { ModelOut } from '../../api/client'

export interface TestConfig {
  selected: string[]
  device: string
  iou: number
  imgsz: string
  conf: number
  showLabels: boolean
}

const IMGSZ_OPTIONS = ['320', '480', '640', '960', '1280']

interface Props {
  models: ModelOut[]
  loading: boolean
  cfg: TestConfig
  set: <K extends keyof TestConfig>(key: K, value: TestConfig[K]) => void
  deviceOptions: { label: string; value: string }[]
  /** A/B mode selects models per-side, so hide the multi-select there. */
  hideModelSelect?: boolean
}

/** Shared left-column controls for every Test mode. */
export default function TestControls({ models, loading, cfg, set, deviceOptions, hideModelSelect }: Props) {
  return (
    <Card withBorder radius="md" padding="md">
      <Stack gap="sm">
        {!hideModelSelect && (
          <>
            <Text fw={600} size="sm">
              모델{' '}
              {cfg.selected.length > 1 && (
                <Badge size="xs" ml={4}>
                  앙상블 {cfg.selected.length}
                </Badge>
              )}
            </Text>
            {loading ? (
              <Loader size="sm" />
            ) : models.length === 0 ? (
              <Text c="dimmed" size="sm">
                이 프로젝트에 모델이 없습니다. 먼저 학습하거나 업로드하세요.
              </Text>
            ) : (
              <Checkbox.Group value={cfg.selected} onChange={(v) => set('selected', v)}>
                <Stack gap={6}>
                  {models.map((m) => (
                    <Checkbox
                      key={m.id}
                      value={m.id}
                      label={
                        <Group gap={6} wrap="nowrap">
                          <Text size="sm">{m.name}</Text>
                          <Badge size="xs" variant="light" color="gray">
                            {m.task}
                          </Badge>
                        </Group>
                      }
                    />
                  ))}
                </Stack>
              </Checkbox.Group>
            )}
          </>
        )}

        <Text fw={600} size="sm" mt={hideModelSelect ? 0 : 'xs'}>
          Device
        </Text>
        <SegmentedControl
          size="xs"
          data={deviceOptions}
          value={cfg.device}
          onChange={(v) => set('device', v)}
          fullWidth
        />

        <Text fw={600} size="sm" mt="xs">
          Confidence <Text span c="dimmed">{cfg.conf.toFixed(2)}</Text>
        </Text>
        <Slider min={0} max={1} step={0.01} value={cfg.conf} onChange={(v) => set('conf', v)} label={(v) => v.toFixed(2)} />

        <Text fw={600} size="sm" mt="xs">
          IoU (WBF) <Text span c="dimmed">{cfg.iou.toFixed(2)}</Text>
        </Text>
        <Slider min={0.1} max={0.95} step={0.05} value={cfg.iou} onChange={(v) => set('iou', v)} label={(v) => v.toFixed(2)} />

        <Select
          mt="xs"
          label="Image size"
          size="xs"
          data={IMGSZ_OPTIONS}
          value={cfg.imgsz}
          onChange={(v) => v && set('imgsz', v)}
          allowDeselect={false}
        />

        <Switch
          mt="xs"
          size="sm"
          label="박스 라벨 표시"
          checked={cfg.showLabels}
          onChange={(e) => set('showLabels', e.currentTarget.checked)}
        />
        <Text c="dimmed" size="xs">
          Confidence는 재추론 없이 즉시 필터링됩니다. Device·IoU·Image size 변경 후엔 다시 실행하세요.
        </Text>
      </Stack>
    </Card>
  )
}

import { Badge, Card, Checkbox, Group, Loader, SegmentedControl, Select, Slider, Stack, Text } from '@mantine/core'
import type { ModelOut } from '../../api/client'

export interface TestConfig {
  selected: string[]
  device: string
  iou: number
  imgsz: string
  conf: number
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

        <Text c="dimmed" size="xs" mt="xs">
          Confidence·IoU는 추론에 반영됩니다. 영상 주석은 이 설정으로 박스를 그리고, 정밀 분석은 이 임계값으로 정답과 비교합니다.
        </Text>
      </Stack>
    </Card>
  )
}
